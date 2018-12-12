import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
import json
import glob

# my modules
from image_helpers import data_dir

temp_directory = os.path.join(os.path.expanduser("~"),
                              "tmp_pytorch/")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# # # CLASSES

class Model(models.ResNet):
    def __init__(self,
                 data_loaders,
                 device=device,
                 continue_training=True,
                 with_one_by_one=False,
                 **kwargs):
        super(Model, self).__init__(models.resnet.BasicBlock,
                                    [3, 4, 6, 3],
                                    **kwargs)
        import torch.utils.model_zoo as model_zoo

        if not continue_training:
            self.load_state_dict(model_zoo.load_url(
                models.resnet.model_urls["resnet34"]
            ))

        # # Change the necessary layer
        # # (must be done _after_ loading pretrained weights)
        if with_one_by_one:
            raise NotImplementedError()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1),)

        num_ftrs = self.fc.in_features
        self.fc = nn.Linear(num_ftrs, 1,)

        # # # Add extra data you want to keep track of

        self.filename_weights = "data/CNN/torch.vanilla.weights"
        self.filename_metadata = "data/CNN/torch.vanilla.metadata.json"
        self.filename_logger = "pytorch.vanilla.log"
        self.filename_batch_logger = "pytorch.vanilla.bybatch.log"


        self.data_loaders = data_loaders
        self.dataset_sizes = {
            key: len(self.data_loaders[key].dataset.samples)
            for key in data_loaders
        }
        self.device = device

        if continue_training:
            self.restore_model_including_metadata()
        else:
            self.epoch_counter = -1
            self.best_validation_loss = np.inf

        if not continue_training:
            if os.path.exists(self.filename_logger):
                filename_logger_tmp = self.filename_logger + ".old"
                os.rename(self.filename_logger, filename_logger_tmp)

            if os.path.exists(self.filename_batch_logger):
                filename_batch_logger_tmp = self.filename_batch_logger + ".old"
                os.rename(self.filename_batch_logger, filename_batch_logger_tmp)

        if not os.path.exists(self.filename_logger):
            with open(self.filename_logger, mode="w") as logger_file:
                print("epoch,loss,val_loss", file=logger_file)

        if not os.path.exists(self.filename_batch_logger):
            with open(self.filename_batch_logger, mode="w") as logger_batch_file:
                print("epoch,batch,epoch_frac,loss", file=logger_batch_file)

    def forward(self, x, verbose=False):
        # like the default `forward`, but just more verbose
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        if verbose:
            print("after layer 1: ", x.shape)

        x = self.layer2(x)
        if verbose:
            print("after layer 2: ", x.shape)

        x = self.layer3(x)
        if verbose:
            print("after layer 3: ", x.shape)

        x = self.layer4(x)
        if verbose:
            print("after layer 4: ", x.shape)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def save_model_metadata(self):
        with open(self.filename_metadata, mode="w") as f:
            metadata = {
                "epoch_counter": self.epoch_counter,
                "best_validation_loss": self.best_validation_loss,
            }

            metadata_json = json.dumps(metadata)

            print(metadata_json, file=f)

    def restore_model_including_metadata(self):
        with open(self.filename_metadata, mode="r") as f:
            metadata = json.loads(f.read())

        self.epoch_counter = metadata["epoch_counter"]
        self.best_validation_loss = metadata["best_validation_loss"]

        self.load_state_dict(torch.load(self.filename_weights))

        return self

    def train_model(self, criterion, optimizer, scheduler,
                    num_epochs=10,
                    verbose=False
                    ):
        """
        criterion: the loss function; callable(prediction, targets)
        optimizer: pytorch optimizer object
        scheduler: LR scheduler (see `torch.optim.lr_scheduler`)

        I guess optimizer already needs to be linked to criterion?
        """
        logger_file = open(self.filename_logger, mode="a")
        logger_batch_file = open(self.filename_batch_logger, mode="a")

        since = time.time()

        best_model_wts = copy.deepcopy(self.state_dict())

        for epoch in range(self.epoch_counter+1, num_epochs):
            self.epoch_counter = epoch
            print(epoch, file=logger_file, end=",")

            # Each epoch has a training and validation phase
            for phase in ["training", "validation"]:
                if phase == "training":
                    scheduler.step(epoch=epoch % scheduler.T_max)
                    self.train()  # Set model to training mode
                else:
                    self.eval()   # Set model to evaluate mode

                running_loss = 0.0

                # Iterate over data.
                num_batches = len(self.data_loaders[phase])
                for idx, data in enumerate(self.data_loaders[phase]):
                    inputs, targets = data
                    inputs = inputs.to(self.device)
                    targets = targets.reshape((-1, 1))
                    targets = targets.to(device=self.device, dtype=torch.float)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == "training"):
                        outputs = self(inputs)
                        if verbose:
                            print("outputs shape: ", outputs.shape)
                        loss = criterion(outputs, targets)

                        # backward + optimize only if in training phase
                        if phase == "training":
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)

                    # update printed stats
                    if phase == "training":
                        print(("Epoch: [{:2d}] "
                               "[batch: {:4d}/{:4d}] "
                               "time: {:4.1f} s, "
                               "batch training loss: {:.8f} dex").format(
                                    epoch,
                                    idx+1,
                                    num_batches,
                                    time.time() - since,
                                    loss.item()**.5,
                                ),
                              end="",
                              )
                        if idx != num_batches-1:
                            print("\r", end="")
                        else:
                            print("")
                        print("{},{},{},{}".format(epoch,
                                                   idx,
                                                   epoch + (idx/num_batches),
                                                   loss.item()**.5),
                              file=logger_batch_file, flush=True)

                epoch_loss = (running_loss / self.dataset_sizes[phase])**.5
                print(epoch_loss, file=logger_file,
                      end=("," if phase == "training" else "\n"))
                if phase == "validation":
                    logger_file.flush()

                print("{:<10} loss: {:.4f} dex".format(
                    phase, epoch_loss))

                # deep copy the model
                if ((phase == "validation") and
                        (epoch_loss < self.best_validation_loss)):
                    self.best_validation_loss = epoch_loss
                    best_model_wts = copy.deepcopy(self.state_dict())
                    torch.save(best_model_wts, self.filename_weights)
                    self.save_model_metadata()

            print()

        time_elapsed = time.time() - since
        print("-" * 20)
        print("Training completed in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60))
        print("Best val loss: {:4f} dex".format(
            self.best_validation_loss))

        # load best model weights
        self.load_state_dict(best_model_wts)
        torch.save(self.state_dict(), self.filename_weights)
        self.save_model_metadata()
        logger_file.close()
        return model

    def apply(self, phase="testing"):
        self.eval()   # Set model to evaluate mode

        # Iterate over data.
        num_batches = len(self.data_loaders[phase])

        targets_list = [None]*num_batches
        outputs_list = [None]*num_batches
        for idx, data in enumerate(self.data_loaders[phase]):
            print("batch: {}/{}".format(idx, num_batches), end="\r")
            inputs, targets = data
            inputs = inputs.to(self.device)
            targets_list[idx] = targets.reshape((-1))

            with torch.set_grad_enabled(False):
                outputs = self(inputs)
            outputs_list[idx] = outputs.reshape((-1))
        print("batch: {}/{}".format(idx, num_batches)) # return cursor to next line


        return np.hstack(targets_list), np.hstack(outputs_list)

# # # FUNCTIONS


def create_pytorch_directory_structure(
    ids_with_images,
    training_ids, validation_ids, testing_ids,
    start_from_scratch=False,
    temp_directory=temp_directory,
    verbose=False,
):
    if start_from_scratch:
        if os.path.isdir(temp_directory):
            shutil.rmtree(temp_directory)

    if not os.path.isdir(temp_directory):
        os.makedirs(temp_directory)

    source_format = os.path.join(
        os.getcwd(),
        data_dir,
        "images",
        "processed",
        "{galaxy_id}.npy")

    target_dir_format = os.path.join(temp_directory, "{phase}", "{galaxy_id}")
    target_format = os.path.join(target_dir_format, "{galaxy_id}.npy")

    phase_ids_with_images = {
        "validation": ids_with_images[np.isin(ids_with_images,
                                              validation_ids)],
        "training": ids_with_images[np.isin(ids_with_images,
                                            training_ids)],
        "testing": ids_with_images[np.isin(ids_with_images,
                                           testing_ids)],
    }

    for phase in ["training", "validation", "testing"]:
        target_pattern = target_format.format(galaxy_id="*", phase=phase)

        already_existing_ids = glob.glob(target_pattern)
        already_existing_ids = [int(os.path.basename(filename).split(".")[0])
                                for filename in already_existing_ids]
        already_existing_ids = set(already_existing_ids)

        tmp_ids_set = set(phase_ids_with_images[phase])

        need_to_make_ids = tmp_ids_set - already_existing_ids

        if not (already_existing_ids <= tmp_ids_set):
            # note: not (a <= b) cannot be swapped with (a>b)!
            raise RuntimeError("The wrong galaxy ids exist in phase directory "
                               "`{}`".format(phase)
                               )

        if verbose:
            num_symlinks_needed = len(need_to_make_ids)
            print("Adding {} symlink{} for {} directory".format(
                        num_symlinks_needed,
                        "" if num_symlinks_needed == 1 else "s",
                        phase),
                  flush=True)

        for i, galaxy_id in enumerate(need_to_make_ids):
            target_dir = target_dir_format.format(galaxy_id=galaxy_id,
                                                  phase=phase)
            if not os.path.isdir(target_dir):
                os.makedirs(target_dir)

            target_filename = target_format.format(galaxy_id=galaxy_id,
                                                   phase=phase)
            os.symlink(
                source_format.format(galaxy_id=galaxy_id),
                target_filename,
            )


def loader(path):
    """expects that `path` points to a `.npy` file"""
    img = np.load(path)
    img = img[1:4]
    if np.random.choice((True, False)):
        img = img[:, :, ::-1]
        img = np.array(img)
    if np.random.choice((True, False)):
        img = img[:, ::-1, :]
        img = np.array(img)

    img = img.transpose((1, 2, 0))  # pytorch is going to rotate it back
    return img


def target_transform(target, classes, df_Y):
    """transforms `target` from a class_index float regression target"""
    target = int(classes[target])
    return df_Y.loc[target].target


def create_data_loaders(
    df_Y,
    temp_directory=temp_directory,
    shuffle=True,
    batch_size=256,
    num_workers=4,
    verbose=False
):
    # Data augmentation and normalization
    data_transforms = {
        "training": transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            # transforms.RandomHorizontalFlip(), # requires a PIL-able image
            # transforms.RandomVerticalFlip(), # requires a PIL-able image
            transforms.ToTensor(),
        ]),
    }
    data_transforms["validation"] = data_transforms["training"]
    data_transforms["testing"] = data_transforms["training"]

    extensions = ["npy"]

    print("Creating `DatasetFolder`s", flush=True)
    image_datasets = {key: datasets.DatasetFolder(os.path.join(temp_directory,
                                                               key),
                                                  loader,
                                                  extensions,
                                                  data_transforms[key],
                                                  # target_transform=target_transform
                                                  )
                      for key in data_transforms}

    print("Fixing `DatasetFolder.samples`", flush=True)
    # overwrite DatasetFolder.samples to use regression targets
    for key in image_datasets:
        new_samples = [None]*len(image_datasets[key])
        for idx, sample in enumerate(image_datasets[key].samples):
            new_samples[idx] = (sample[0],
                                target_transform(sample[1],
                                                 image_datasets[key].classes,
                                                 df_Y),
                                )
        image_datasets[key].samples = new_samples

    print("Creating `DataLoader`s", flush=True)
    data_loaders = {key: torch.utils.data.DataLoader(image_datasets[key],
                                                     batch_size=batch_size,
                                                     shuffle=shuffle,
                                                     num_workers=num_workers)
                    for key in image_datasets}

    if verbose:
        dataset_sizes = {key: len(image_datasets[key])
                         for key in image_datasets}

        print("dataset_sizes: ", dataset_sizes)
        ds = image_datasets["training"]
        print("len training dataset: ", len(ds))
        sample = ds.__getitem__(1)
        print("sample input data shape: ", sample[0].shape)  # image
        print("sample target: ", sample[1])

    return data_loaders

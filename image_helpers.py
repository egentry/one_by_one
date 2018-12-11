import astropy
from astropy import wcs
from astropy.nddata import Cutout2D
import numpy as np
import os
import urllib
import warnings

raw_data_dir = "/Volumes/G-RAID Studio/egentry/one_by_one/data/images/full"
data_dir = "data"
bands = ("u", "g", "r", "i", "z")


def get_raw_filename(run, camcol, field, band, raw_data_dir=raw_data_dir):
    output_filename = os.path.join(raw_data_dir,
                                   ("{run}-"
                                    "{camcol}-"
                                    "{field}-"
                                    "{band}.fits.bz2").format(
                                        run=run,
                                        camcol=camcol,
                                        field=field,
                                        band=band,
                                        )
                                   )
    return output_filename


def get_cutout_filename(galaxy_id, band, data_dir=data_dir):
    cutout_filename = os.path.join(data_dir,
                                   "images",
                                   "cutout",
                                   "{galaxy_id}",
                                   "{galaxy_id}-{band}.fits",
                                   ) \
                              .format(
                                    galaxy_id=galaxy_id,
                                    band=band
                                )
    return cutout_filename


def download_image(run, camcol, field, band, output_filename, overwrite=False):
    if (raw_data_dir in output_filename) and \
            (not os.path.exists(raw_data_dir)):
        raise OSError("dir not found; is drive mounted? (dir={})".format(
            raw_data_dir)
        )

    url_base = ("http://data.sdss.org/sas/dr14/eboss/photoObj/frames/301/"
                "{run}/{camcol}/frame-{band}-{run:>06d}-{camcol}-{field:>04d}"
                ".fits.bz2")

    if (not os.path.exists(output_filename)) or overwrite:
        try:
            data = urllib.request \
                         .urlopen(url_base.format(run=run, camcol=camcol,
                                                  field=field, band=band,)) \
                         .read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                warnings.warn("Missing remote image file: {}-{}-{}-{}".format(
                    run, camcol, field, band,
                    ))
                return
            else:
                raise e

        with open(output_filename, "wb") as f:
            f.write(data)


def get_cutout(hdu, ra, dec, size_arcsec=38):
    old_header = hdu.header.copy()
    w = wcs.WCS(hdu.header)

    target_coords = astropy.coordinates.SkyCoord(ra, dec,
                                                 unit="deg",
                                                 # frame="icrs",
                                                 )
    cutout_size = int(size_arcsec / .4)  # .4 arcsec scale

    # check if the image is going to be cutoff because the galaxy
    # is too close to an edge.
    # In practice we don't need to do anything now;
    # we'll just filter out the cutoff images later.
    cutoff = False
    target_coords_pixel = target_coords.to_pixel(w)
    if (old_header["NAXIS1"] < target_coords_pixel[0] + cutout_size/2):
        cutoff = True
    if (old_header["NAXIS2"] < target_coords_pixel[1] + cutout_size/2):
        cutoff = True
    if (0 > target_coords_pixel[0] - cutout_size/2):
        cutoff = True
    if (0 > target_coords_pixel[1] - cutout_size/2):
        cutoff = True

    c = Cutout2D(hdu.data,
                 position=target_coords,
                 size=int(np.ceil(size_arcsec / .4)),
                 wcs=w)

    hdu = astropy.io.fits.PrimaryHDU(data=c.data,
                                     header=c.wcs.to_header())

    # copy all the old fields to the new header
    # this _should_ update wcs, but won't strip now-unneeded fields
    old_header.update(hdu.header)
    hdu.header.update(old_header)

    if (not cutoff) and (hdu.data.shape != (cutout_size, cutout_size)):
        print(old_header["RUN"], old_header["CAMCOL"], old_header["FILTER"])
        raise RuntimeError(("image shape ({0}) "
                            "does not match desired dimensions"
                            "(({1},{1})").format(
            hdu.data.shape, cutout_size))

    return hdu

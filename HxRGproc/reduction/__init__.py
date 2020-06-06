""" This module is to reduce the output of Teledyne HxRG detectors """
from .reduction import remove_biases_in_cube, subtract_reference_pixels, remove_bias_preserve_pedestal_in_cube, slope_img_from_cube
from .reduction import apply_nonlinearcorr_polynomial
from .generate_slope_images import load_data_cube

import warnings

from .models.mtcnn import (MTCNN, ONet, PNet, RNet,
                           fixed_image_standardization, prewhiten)
from .models.utils.detect_face import extract_face

warnings.filterwarnings(
    action="ignore", 
    message="This overload of nonzero is deprecated:\n\tnonzero()", 
    category=UserWarning
)
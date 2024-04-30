
# Encoding_face_Comparison

Face comparison based on 128 dimension facial encodings. Face_encodings method returns an encoding of the input image. Then, the compare_faces method compares the encodings through a distance parameter to see if there is a match. Then, the encoding with the least distance gets selected since it’s the closest match. After getting the match, the image’s title is retrieved using the image’s index in the list.


## Installation

Install libraries with pip

```bash
  pip install opencv
  pip install face_recognition
  pip install pickle
```
    
## Running Tests

To run tests, run the following command

```bash
  python Data_collect.python
  python Data_check.py
```


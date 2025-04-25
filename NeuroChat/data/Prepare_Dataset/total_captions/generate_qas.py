import pandas as pd
import random

# Load CSV files
df_train = pd.read_csv("/mnt/data/train_labels.csv")
df_btcmri = pd.read_csv("/mnt/data/test_labels_btcmri.csv")
df_btmri = pd.read_csv("/mnt/data/test_labels_btmri.csv")

# Define question and answer variations

# Tumor presence
tumor_presence_qs = [
    "Is there a tumor visible in this scan?",
    "Does this MRI image show a tumor?",
    "Can a tumor be observed in the image?",
    "Is the scan indicating the presence of a tumor?",
    "Does this brain image contain a tumor?",
    "Is this an example of a tumorous brain scan?",
    "Is a tumor present in the scan provided?",
    "Is there any abnormal growth in this MRI?",
    "Can you detect any tumor in this image?",
    "Does this MRI show signs of a tumor?"
]

tumor_presence_as = {
    'yes': [
        "Yes, a tumor is present.",
        "A tumor is visible in the image.",
        "Indeed, the scan shows a tumor.",
        "Yes, it looks like there is a tumor.",
        "Yes, abnormal tissue is seen.",
        "A tumor can be seen clearly.",
        "Confirmed, tumor is present.",
        "Yes, this is a positive case for tumor.",
        "Yes, tumor identified.",
        "Certainly, a tumor exists."
    ],
    'no': [
        "No, there is no tumor present.",
        "The image does not show a tumor.",
        "No tumor is detected here.",
        "The scan appears normal.",
        "There’s no abnormality in the image.",
        "No tumor can be seen.",
        "The image is tumor-free.",
        "No indication of tumor.",
        "No growth is present.",
        "Tumor not found in this image."
    ]
}

# Tumor type
tumor_type_qs = [
    "What kind of tumor is present in the image?",
    "Identify the tumor type in this scan.",
    "Which tumor type does this image show?",
    "Can you tell the type of tumor shown?",
    "Specify the tumor type visible here.",
    "What is the classification of this tumor?",
    "What tumor subtype is this?",
    "What does the scan indicate as tumor type?",
    "What is the detected tumor type?",
    "What specific tumor is present here?"
]

tumor_type_as_template = {
    '-': ["No tumor present."],
    'glioma': [
        "It is a glioma.",
        "The tumor is glioma.",
        "Glioma tumor is shown here.",
        "This image displays a glioma tumor.",
        "A glioma is present in this scan.",
        "Detected tumor: glioma.",
        "This is classified as a glioma.",
        "Glioma is the tumor type here.",
        "This scan has glioma type.",
        "Type of tumor: glioma."
    ],
    'meningioma': [
        "It is a meningioma.",
        "The tumor is meningioma.",
        "Meningioma tumor is shown here.",
        "This image displays a meningioma tumor.",
        "A meningioma is present in this scan.",
        "Detected tumor: meningioma.",
        "This is classified as a meningioma.",
        "Meningioma is the tumor type here.",
        "This scan has meningioma type.",
        "Type of tumor: meningioma."
    ],
    'pituitary': [
        "It is a pituitary tumor.",
        "The tumor is pituitary.",
        "Pituitary tumor is shown here.",
        "This image displays a pituitary tumor.",
        "A pituitary is present in this scan.",
        "Detected tumor: pituitary.",
        "This is classified as a pituitary tumor.",
        "Pituitary is the tumor type here.",
        "This scan has pituitary type.",
        "Type of tumor: pituitary."
    ]
}

# Image view type
image_type_qs = [
    "What is the view type of this image?",
    "Which plane is shown in the image?",
    "Identify the image orientation.",
    "In which direction is this MRI taken?",
    "What kind of image view is this?",
    "Can you name the scan view type?",
    "What is the imaging perspective?",
    "Which anatomical plane is used?",
    "What's the viewing angle here?",
    "What image type is displayed here?"
]

image_type_as_template = {
    'axial': [
        "This is an axial view.",
        "The image is axial.",
        "Axial plane is shown.",
        "Scan is from axial direction.",
        "Axial perspective image.",
        "View type: axial.",
        "Orientation: axial.",
        "This scan uses axial plane.",
        "Axial slice is shown.",
        "Axial type image."
    ],
    'sagittal': [
        "This is a sagittal view.",
        "The image is sagittal.",
        "Sagittal plane is shown.",
        "Scan is from sagittal direction.",
        "Sagittal perspective image.",
        "View type: sagittal.",
        "Orientation: sagittal.",
        "This scan uses sagittal plane.",
        "Sagittal slice is shown.",
        "Sagittal type image."
    ],
    'coronal': [
        "This is a coronal view.",
        "The image is coronal.",
        "Coronal plane is shown.",
        "Scan is from coronal direction.",
        "Coronal perspective image.",
        "View type: coronal.",
        "Orientation: coronal.",
        "This scan uses coronal plane.",
        "Coronal slice is shown.",
        "Coronal type image."
    ]
}

# Function to generate a triplet per image
def generate_triplets(df):
    triplet_data = []
    for idx, row in df.iterrows():
        tumor = row['tumor']
        tumor_type = row['tumor_type']
        image_type = row['image_type']
        image_name = row['image']

        q1 = random.choice(tumor_presence_qs)
        a1 = random.choice(tumor_presence_as[tumor])
        q2 = random.choice(tumor_type_qs)
        a2 = random.choice(tumor_type_as_template[tumor_type])
        q3 = random.choice(image_type_qs)
        a3 = random.choice(image_type_as_template[image_type])

        triplet_data.append({
            "image": image_name,
            "question_1": q1,
            "answer_1": a1,
            "question_2": q2,
            "answer_2": a2,
            "question_3": q3,
            "answer_3": a3
        })
    return pd.DataFrame(triplet_data)

# Generate triplet tables
triplet_train = generate_triplets(df_train)
triplet_btcmri = generate_triplets(df_btcmri)
triplet_btmri = generate_triplets(df_btmri)

# Optional: Save to CSVs if needed
# triplet_train.to_csv("train_triplets.csv", index=False)
# triplet_btcmri.to_csv("btcmri_triplets.csv", index=False)
# triplet_btmri.to_csv("btmri_triplets.csv", index=False)

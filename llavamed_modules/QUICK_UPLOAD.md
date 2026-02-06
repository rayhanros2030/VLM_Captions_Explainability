# Quick Manual Upload - Copy & Paste

## On Lambda (SSH in first):

```bash
ssh ubuntu@YOUR_LAMBDA_IP

# Create folders
mkdir -p /home/ubuntu/projects/llavamed-integration
mkdir -p /home/ubuntu/data/histopathology_dataset/images
mkdir -p /home/ubuntu/data/histopathology_outputs
```

## On Windows (PowerShell):

```powershell
# Set your Lambda IP
$LAMBDA_IP = "YOUR_LAMBDA_IP"  # <-- CHANGE THIS
$LAMBDA_USER = "ubuntu"

# Upload modules folder
scp -r "C:\Users\PC\Downloads\llavamed_modules" "${LAMBDA_USER}@${LAMBDA_IP}:/home/ubuntu/projects/llavamed-integration/"

# Upload CSV file
scp "C:\Users\PC\Downloads\patchgastricadc222_prepared\dataset_with_labels.csv" "${LAMBDA_USER}@${LAMBDA_IP}:/home/ubuntu/data/histopathology_dataset/dataset_with_labels.csv"
```

## That's it! 

Images can be uploaded later if needed. The system will work without them for testing.



# Manual Upload Guide - Lambda Labs Setup

## Step 1: Create Folders on Lambda

SSH into your Lambda instance and create these folders:

```bash
ssh ubuntu@YOUR_LAMBDA_IP

# Create project directory
mkdir -p /home/ubuntu/projects/llavamed-integration

# Create data directories
mkdir -p /home/ubuntu/data/histopathology_dataset/images
mkdir -p /home/ubuntu/data/histopathology_outputs
```

## Step 2: Upload Files from Windows

### Option A: Using SCP (Command Line)

Open PowerShell on Windows and run these commands (replace `YOUR_LAMBDA_IP`):

```powershell
$LAMBDA_IP = "YOUR_LAMBDA_IP"  # Replace with your Lambda IP
$LAMBDA_USER = "ubuntu"

# 1. Upload the entire llavamed_modules folder
scp -r "C:\Users\PC\Downloads\llavamed_modules" "${LAMBDA_USER}@${LAMBDA_IP}:/home/ubuntu/projects/llavamed-integration/"

# 2. Upload the dataset CSV file
scp "C:\Users\PC\Downloads\patchgastricadc222_prepared\dataset_with_labels.csv" "${LAMBDA_USER}@${LAMBDA_IP}:/home/ubuntu/data/histopathology_dataset/dataset_with_labels.csv"
```

### Option B: Using WinSCP or FileZilla (GUI)

1. **Connect to Lambda:**
   - Host: `YOUR_LAMBDA_IP`
   - Username: `ubuntu`
   - Protocol: SFTP

2. **Upload llavamed_modules folder:**
   - Local: `C:\Users\PC\Downloads\llavamed_modules`
   - Remote: `/home/ubuntu/projects/llavamed-integration/llavamed_modules`
   - Drag and drop the entire folder

3. **Upload dataset CSV:**
   - Local: `C:\Users\PC\Downloads\patchgastricadc222_prepared\dataset_with_labels.csv`
   - Remote: `/home/ubuntu/data/histopathology_dataset/dataset_with_labels.csv`

## Step 3: Upload Images (Later - Optional)

The images folder has ~262k files. You can upload this later or use a different method:

**Option 1: Upload later via SCP**
```powershell
scp -r "C:\Users\PC\Downloads\extracted_archive\patches_captions\patches_captions\*" "${LAMBDA_USER}@${LAMBDA_IP}:/home/ubuntu/data/histopathology_dataset/images/"
```

**Option 2: Compress first, then upload**
```powershell
# On Windows: Compress the images folder first
# Then upload the zip file
scp "C:\Users\PC\Downloads\images.zip" "${LAMBDA_USER}@${LAMBDA_IP}:/home/ubuntu/data/"

# On Lambda: Extract
ssh ubuntu@YOUR_LAMBDA_IP
cd /home/ubuntu/data
unzip images.zip -d histopathology_dataset/images/
```

## Step 4: Verify Upload

On Lambda, check that files are there:

```bash
# Check modules
ls -la /home/ubuntu/projects/llavamed-integration/llavamed_modules/

# Check CSV
ls -lh /home/ubuntu/data/histopathology_dataset/dataset_with_labels.csv

# Check images (if uploaded)
ls /home/ubuntu/data/histopathology_dataset/images/ | head -5
```

## Summary: What to Upload

| What to Upload | Local Path | Remote Path |
|----------------|------------|-------------|
| **llavamed_modules folder** | `C:\Users\PC\Downloads\llavamed_modules` | `/home/ubuntu/projects/llavamed-integration/llavamed_modules/` |
| **dataset_with_labels.csv** | `C:\Users\PC\Downloads\patchgastricadc222_prepared\dataset_with_labels.csv` | `/home/ubuntu/data/histopathology_dataset/dataset_with_labels.csv` |
| **Images folder** (optional, later) | `C:\Users\PC\Downloads\extracted_archive\patches_captions\patches_captions\*` | `/home/ubuntu/data/histopathology_dataset/images/` |

## Next Steps After Upload

Once files are uploaded, run the setup script:

```bash
cd /home/ubuntu/projects/llavamed-integration
chmod +x llavamed_modules/lambda_setup.sh
./llavamed_modules/lambda_setup.sh
```



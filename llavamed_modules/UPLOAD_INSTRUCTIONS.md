# How to Upload Files to Lambda

## Quick Method: Copy-Paste Commands

Open PowerShell and run these commands one by one. **Replace `YOUR_LAMBDA_IP` with your actual Lambda IP address.**

```powershell
# Set your Lambda IP (CHANGE THIS!)
$LAMBDA_IP = "YOUR_LAMBDA_IP"
$LAMBDA_USER = "ubuntu"

# Step 1: Upload modules
Write-Host "Uploading modules..." -ForegroundColor Yellow
scp -r "C:\Users\PC\Downloads\llavamed_modules" "${LAMBDA_USER}@${LAMBDA_IP}:/home/ubuntu/projects/llavamed-integration/"

# Step 2: Create directories on Lambda
Write-Host "Creating directories..." -ForegroundColor Yellow
ssh "${LAMBDA_USER}@${LAMBDA_IP}" "mkdir -p /home/ubuntu/data/histopathology_dataset/images /home/ubuntu/data/histopathology_outputs"

# Step 3: Upload dataset CSV
Write-Host "Uploading dataset CSV..." -ForegroundColor Yellow
scp "C:\Users\PC\Downloads\patchgastricadc222_prepared\dataset_with_labels.csv" "${LAMBDA_USER}@${LAMBDA_IP}:/home/ubuntu/data/histopathology_dataset/dataset_with_labels.csv"

# Step 4: Images (OPTIONAL - can skip and do later)
# This will take a LONG time (~262k files)
Write-Host "Upload images? This will take hours. Skip for now? (yes/no)" -ForegroundColor Yellow
$upload = Read-Host
if ($upload -ne "yes") {
    Write-Host "Skipping images. Upload later with:" -ForegroundColor Yellow
    Write-Host "scp -r `"C:\Users\PC\Downloads\extracted_archive\patches_captions\patches_captions\*`" ${LAMBDA_USER}@${LAMBDA_IP}:/home/ubuntu/data/histopathology_dataset/images/" -ForegroundColor White
} else {
    Write-Host "Uploading images (this will take a while)..." -ForegroundColor Yellow
    scp -r "C:\Users\PC\Downloads\extracted_archive\patches_captions\patches_captions\*" "${LAMBDA_USER}@${LAMBDA_IP}:/home/ubuntu/data/histopathology_dataset/images/"
}
```

## Alternative: Run Script Directly

If the script opens in a notebook, run it from PowerShell like this:

```powershell
# Navigate to the directory
cd C:\Users\PC\Downloads\llavamed_modules

# Run with PowerShell explicitly
powershell.exe -ExecutionPolicy Bypass -File .\upload_to_lambda_direct.ps1
```

**But first**, edit `upload_to_lambda_direct.ps1` and change `YOUR_LAMBDA_IP` to your actual IP.

## What Gets Uploaded

1. ✅ **llavamed_modules/** → `/home/ubuntu/projects/llavamed-integration/llavamed_modules/`
2. ✅ **dataset_with_labels.csv** → `/home/ubuntu/data/histopathology_dataset/dataset_with_labels.csv`
3. ⏭️ **Images** (optional, can do later) → `/home/ubuntu/data/histopathology_dataset/images/`

## After Upload

SSH into Lambda and run setup:

```bash
ssh ubuntu@YOUR_LAMBDA_IP
cd /home/ubuntu/projects/llavamed-integration
chmod +x llavamed_modules/lambda_setup.sh
./llavamed_modules/lambda_setup.sh
```



# Upload Images to Lambda

You have ~262,000 image files to upload. Here are your options:

## Option 1: Compress First (RECOMMENDED - Much Faster)

Compressing the images first will make the upload much faster.

### Step 1: Compress on Windows

Open PowerShell and run:

```powershell
# Navigate to the images directory
cd "C:\Users\PC\Downloads\extracted_archive\patches_captions"

# Compress the patches_captions folder (this creates a zip file)
Compress-Archive -Path "patches_captions\*" -DestinationPath "C:\Users\PC\Downloads\images.zip" -CompressionLevel Optimal

# This will take a while (10-30 minutes depending on your system)
# The zip file will be large (~several GB)
```

### Step 2: Upload the zip file

```powershell
$LAMBDA_IP = "YOUR_LAMBDA_IP"  # Replace with your Lambda IP
$LAMBDA_USER = "ubuntu"

# Upload the zip file (much faster than individual files)
scp "C:\Users\PC\Downloads\images.zip" "${LAMBDA_USER}@${LAMBDA_IP}:/home/ubuntu/data/"
```

### Step 3: Extract on Lambda

SSH into Lambda and extract:

```bash
ssh ubuntu@YOUR_LAMBDA_IP

# Extract images
cd /home/ubuntu/data
unzip images.zip -d histopathology_dataset/images/

# Clean up the zip file to save space
rm images.zip
```

## Option 2: Direct Upload (Slower but Simpler)

If you prefer to upload directly without compression:

```powershell
$LAMBDA_IP = "YOUR_LAMBDA_IP"  # Replace with your Lambda IP
$LAMBDA_USER = "ubuntu"

# Upload all images (this will take HOURS - maybe 4-8 hours)
scp -r "C:\Users\PC\Downloads\extracted_archive\patches_captions\patches_captions\*" "${LAMBDA_USER}@${LAMBDA_IP}:/home/ubuntu/data/histopathology_dataset/images/"
```

**Note:** This will take a very long time. You can run it in the background or use `nohup` if you want to disconnect.

## Option 3: Upload in Batches (If Connection Drops)

If your connection is unstable, you can upload in smaller batches:

```powershell
$LAMBDA_IP = "YOUR_LAMBDA_IP"
$LAMBDA_USER = "ubuntu"
$sourceDir = "C:\Users\PC\Downloads\extracted_archive\patches_captions\patches_captions"

# Get first 1000 files
Get-ChildItem $sourceDir -File | Select-Object -First 1000 | ForEach-Object {
    scp $_.FullName "${LAMBDA_USER}@${LAMBDA_IP}:/home/ubuntu/data/histopathology_dataset/images/"
}
```

## Recommended: Use Compression (Option 1)

Compression is much faster because:
- Single file transfer instead of 262k individual files
- Compressed data is smaller
- More reliable (less chance of connection issues)

## After Upload: Verify

On Lambda, check images are there:

```bash
# Count images
find /home/ubuntu/data/histopathology_dataset/images -type f | wc -l

# Should show ~262000 files
```



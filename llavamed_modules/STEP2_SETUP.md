# Step 2: Setup on Lambda

## Verify Upload First

SSH into Lambda and check files are there:

```bash
ssh ubuntu@YOUR_LAMBDA_IP

# Check modules
ls -la /home/ubuntu/projects/llavamed-integration/llavamed_modules/

# Check CSV
ls -lh /home/ubuntu/data/histopathology_dataset/dataset_with_labels.csv
```

## Run Setup Script

```bash
cd /home/ubuntu/projects/llavamed-integration

# Make setup script executable
chmod +x llavamed_modules/lambda_setup.sh

# Run setup (this installs dependencies, checks GPU, etc.)
./llavamed_modules/lambda_setup.sh
```

This will:
- Install Python dependencies
- Download NLTK data
- Verify GPU is available
- Set up PYTHONPATH
- Check for LLaVA-Med repository

## After Setup Completes

You're ready to run the pipeline! See STEP3_RUN.md for next steps.



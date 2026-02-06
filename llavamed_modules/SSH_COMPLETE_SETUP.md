# Complete SSH Setup Guide for Lambda Labs

## Step 1: Find Your SSH Key Files

Your key files are named `llavamed_modules` and `llavamed_modules.pub`. Let's find them:

### Option A: Check Downloads folder
The files are likely in: `C:\Users\PC\Downloads\llavamed_modules\`

### Option B: Search for them
Open PowerShell and run:
```powershell
Get-ChildItem -Path C:\Users\PC -Recurse -Filter "llavamed_modules.pub" -ErrorAction SilentlyContinue | Select-Object FullName
```

This will show you where the file is.

## Step 2: Get Your Public Key

Once you find the location, get your public key:

```powershell
# Replace with the actual path if different
Get-Content "C:\Users\PC\Downloads\llavamed_modules\llavamed_modules.pub"
```

**Copy the entire output** - it should look like:
```
ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQ... (long string) ... your_email@example.com
```

## Step 3: Add Public Key to Lambda Labs

1. **Go to Lambda Labs**: https://lambdalabs.com/
2. **Log in** to your account
3. **Click on your profile/account** (usually top right)
4. **Go to "SSH Keys"** or "Settings" → "SSH Keys"
5. **Click "Add SSH Key"** or "New Key"
6. **Paste your public key** (the entire string you copied)
7. **Give it a name** (e.g., "Windows PC")
8. **Click "Save"** or "Add"

## Step 4: Connect to Lambda

Once the key is added, connect using the full path to your private key:

```powershell
# Navigate to where your key is (or use full path)
cd C:\Users\PC\Downloads\llavamed_modules

# Connect to Lambda
ssh -i llavamed_modules ubuntu@150.136.219.166
```

Or use the full path:
```powershell
ssh -i "C:\Users\PC\Downloads\llavamed_modules\llavamed_modules" ubuntu@150.136.219.166
```

## Step 5: First Connection

The first time you connect, you'll see:
```
The authenticity of host '150.136.219.166' can't be established.
Are you sure you want to continue connecting (yes/no/[fingerprint])?
```

Type `yes` and press Enter.

## Troubleshooting

**If you get "Permission denied":**
- Make sure you added the **public** key (`.pub` file) to Lambda Labs
- Wait 1-2 minutes after adding the key for it to sync
- Make sure you're using the correct path to your private key

**If you can't find your key files:**
- They might be in the directory where you ran `ssh-keygen`
- Check your current directory: `pwd` (if you're in PowerShell, check with `Get-Location`)

**If the key path is wrong:**
- Use the full absolute path: `C:\Users\PC\Downloads\llavamed_modules\llavamed_modules`
- Make sure there are no spaces in the path (use quotes if needed)



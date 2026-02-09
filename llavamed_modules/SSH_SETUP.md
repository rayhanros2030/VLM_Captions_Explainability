# SSH Key Setup for Lambda Labs

## Option 1: Generate SSH Key on Windows (Recommended)

### Step 1: Check if you already have SSH keys

Open PowerShell and run:
```powershell
ls ~/.ssh/id_rsa.pub
```

If the file exists, you already have a key. Skip to Step 3.

### Step 2: Generate a new SSH key

```powershell
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
```

- Press Enter to accept default location (`C:\Users\PC\.ssh\id_rsa`)
- Enter a passphrase (optional, but recommended) or press Enter for no passphrase
- Press Enter again to confirm

### Step 3: Get your public key

```powershell
cat ~/.ssh/id_rsa.pub
```

Copy the entire output (starts with `ssh-rsa` and ends with your email).

### Step 4: Add key to Lambda Labs

1. Go to Lambda Labs dashboard: https://lambdalabs.com/
2. Log in to your account
3. Go to **SSH Keys** section (usually in Settings or Account)
4. Click **Add SSH Key** or **New Key**
5. Paste your public key
6. Give it a name (e.g., "My Windows PC")
7. Save

### Step 5: Try SSH again

```powershell
ssh ubuntu@150.136.219.166
```

## Option 2: Use Lambda Labs Web Terminal

If SSH keys are taking too long:

1. Go to Lambda Labs dashboard
2. Find your instance
3. Look for **"Web Terminal"** or **"Console"** button
4. Click it to open a browser-based terminal
5. You can run commands directly there

## Option 3: Download SSH Key from Lambda

Some Lambda setups provide a key file:

1. In Lambda Labs dashboard, find your instance
2. Look for **"Download SSH Key"** or **"Access"** section
3. Download the key file (usually `.pem` file)
4. Use it to connect:

```powershell
ssh -i path/to/downloaded-key.pem ubuntu@150.136.219.166
```

## Troubleshooting

**If you get "Permission denied" after adding key:**
- Make sure you copied the **public** key (id_rsa.pub), not the private key
- Wait a minute for Lambda to update
- Try: `ssh -v ubuntu@150.136.219.166` to see detailed error messages

**If key generation fails:**
- Make sure OpenSSH is installed (Windows 10+ should have it)
- Try: `Get-WindowsCapability -Online | Where-Object Name -like 'OpenSSH*'`



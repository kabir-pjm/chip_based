@echo off
cd /d c:\Users\LENOVO\Downloads\chipbased
echo === Step 1: Git Init ===
git init
echo === Step 2: Add All ===
git add -A
echo === Step 3: Commit ===
git commit -m "Initial commit: chip-based thermal governor project"
echo === Step 4: Branch ===
git branch -M main
echo === Step 5: Remote ===
git remote add origin https://github.com/kabir-pjm/chip_based.git
echo === Step 6: Push ===
git push -u origin main --force
echo === Done ===

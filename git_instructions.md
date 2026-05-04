You accidentally committed the runs/ folder because it wasn't in a .gitignore file!

Since you've already created the commit, the cleanest way to fix this is to remove the folder from git's memory (without deleting the actual files from your computer), tell git to ignore it in the future, and then amend your last commit.

Run these three commands in your terminal:

1. Remove the folder from the commit:
git rm -r --cached runs/

2. Add the folder to your .gitignore so it never gets committed again:
echo "runs/" >> .gitignore
git add .gitignore

3. Amend your previous commit (this updates the commit you just made, keeping your original message):
git commit --amend --no-edit

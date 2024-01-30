cd ../..
git add `find . \( -name "*.py" -o -name "*.sh" -o -name "*.m" -o -name "*.md" -o -name "LICENCE" \)`
git commit -vm "$1"
git push -u origin master
echo "yeah" 
jupyter nbconvert notebooks/introduction.ipynb --to markdown --output-dir docs/tutorials
mkdocs gh-deploy
rm -r docs/tutorials/introduction*

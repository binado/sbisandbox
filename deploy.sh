jupyter nbconvert notebooks/introduction.ipynb --to markdown --output-dir docs/tutorials
jupyter nbconvert notebooks/twomoons.ipynb --to markdown --output-dir docs/tutorials
mkdocs gh-deploy
rm -r docs/tutorials/introduction*
rm -r docs/tutorials/twomoons*

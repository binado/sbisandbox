jupyter nbconvert notebooks/introduction.ipynb --to markdown --output-dir docs/tutorials
jupyter nbconvert notebooks/twomoons.ipynb --to markdown --output-dir docs/tutorials
jupyter nbconvert notebooks/mcmc.ipynb --to markdown --output-dir docs/tutorials
mkdocs gh-deploy
rm -r docs/tutorials/introduction*
rm -r docs/tutorials/twomoons*
rm -r docs/tutorials/mcmc*

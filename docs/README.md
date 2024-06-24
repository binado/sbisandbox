# Project documentation

We use [MKDocs](https://www.mkdocs.org/) and [`mkdocs-material`](https://squidfunk.github.io/mkdocs-material/) to build a static website hosted in Github Pages.

## Deploying to Github Pages

The `deploy.sh` script converts all Jupyter notebooks in the repo into markdown files in the `/docs` subdirectory, and deploys with the built-in command from MKDocs

``` bash
mkdocs gh-deploy
```

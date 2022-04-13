This folder includes several folders as follows:

- `in` stores inputs for the `interpret.ipynb`
- `dict` stores saved plots from the `interpret.ipynb`
- `out` stores saved outputs (plots + dictionary) from `evaluator.py`
- `all` is an auxillary folder for online evaluations, where matrices are downloaded and removed as they are used.
- `slurms` has the output files downloaded from cluster. You can parse these using the two regular expressions eblow:
  - `((.*[\n|\r|\r\n]){3})Success` to get the lines related to succesful runs on matrices.
  - `\/MM\/([_\-a-zA-Z0-9]+\/[_\-a-zA-Z0-9]+)\.tar\.gz` to get the matrix `GROUP/NAME` pairs, which are stored in index files.

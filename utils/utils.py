import os
def list_files(root, suffix, prefix=False):
    root = os.path.expanduser(root)
    files = list(
        filter(
            lambda p: os.path.isfile(os.path.join(root, p)) and p.endswith(suffix),
            os.listdir(root)
        )
    )
    if prefix is True:
        files = [os.path.join(root, d) for d in files]
    return files
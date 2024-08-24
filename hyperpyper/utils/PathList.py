from pathlib import Path
import pandas as pd


class PathList:
    """
    A class that holds a list of pathlib.Path objects and allows various operations on them.
    """
    def __init__(self, paths=None):
        """
        Initialize a PathList object.

        Args:
            paths (list of str or list of pathlib.Path, optional): List of paths to initialize the PathList with.
        """
        if paths is None:
            self.paths = []
        else:
            self.paths = [Path(path) if isinstance(path, str) else path for path in paths]


    def __add__(self, other):
        """
        Concatenate two PathList objects.

        Args:
            other (PathList): Another PathList object to concatenate with.

        Returns:
            PathList: A new PathList object with concatenated paths.
        """
        if isinstance(other, PathList):
            return PathList(self.paths + other.paths)
        else:
            raise TypeError(f"Unsupported type for concatenation: {type(other)}")


    def __sub__(self, other):
        """
        Remove the specified path items from the list.        
        """
        if isinstance(other, str) or isinstance(other, Path) or isinstance(other, PathList) or isinstance(other, list):
            return self.remove(other)
        else:
             raise TypeError(f"Unsupported type for path subtraction: {type(other)}")


    def __truediv__(self, other):
        """
        Concatenate paths in the PathList with another path.

        Args:
            other (str, pathlib.Path, or PathList): The path or PathList to concatenate with.

        Returns:
            PathList: A new PathList with concatenated paths.
        """
        new_pathlist = PathList(self.paths)

        if isinstance(other, str) or isinstance(other, Path):
            new_pathlist.paths = [path / other for path in new_pathlist.paths]
        else:
            raise TypeError(f"Unsupported type for path concatenation: {type(other)}")

        return new_pathlist


    def __rtruediv__(self, other):
        """
        Add a prefix to all paths in the PathList.

        Args:
            other (str or pathlib.Path): The prefix to add.

        Returns:
            PathList: A new PathList with the prefix added to all paths.
        """
        new_pathlist = PathList(self.paths)

        if isinstance(other, str) or isinstance(other, Path):
            new_pathlist.paths = [other / path for path in new_pathlist.paths]
        else:
            raise TypeError(f"Unsupported type for path concatenation: {type(other)}")

        return new_pathlist


    def __str__(self):
        """
        Get a string representation of all paths in the PathList.

        Returns:
            list of str: List of string representations of paths.
        """
        return [str(path) for path in self.paths]


    def __repr__(self):
        """
        Get a string representation of the PathList.

        Returns:
            str: String representation of the PathList object.
        """
        return f'PathList({self.paths})'


    def __len__(self):
        return len(self.paths)


    def __getitem__(self, index):
        """
        Get a path at a specific index in the PathList.

        Args:
            index (int): Index of the path to retrieve.

        Returns:
            Path: The path at the specified index.
        """
        return self.paths[index]


    def append(self, path):
        """
        Append a path to the PathList.

        Args:
            path (str or pathlib.Path): The path to append.
        """

        if isinstance(path, str):
            self.paths.append(Path(path))
        elif isinstance(path, Path):
            self.paths.append(path)
        else:
            raise TypeError(f"Unsupported type for path appending: {type(other)}")


    def find_replace(self, find, replace):
        """
        Perform find/replace operations on all paths in the PathList.

        Args:
            find (str or pathlib.Path): The string or path to find.
            replace (str or pathlib.Path): The string or path to replace with.
        """
        if isinstance(find, str) and isinstance(replace, str):
            self.paths = [Path(str(path).replace(find, replace)) for path in self.paths]
        elif isinstance(find, Path) and isinstance(replace, str):
            self.paths = [Path(str(path).replace(str(find), replace)) for path in self.paths]
        elif isinstance(find, str) and isinstance(replace, Path):
            self.paths = [Path(str(path).replace(find, str(replace))) for path in self.paths]
        elif isinstance(find, Path) and isinstance(replace, Path):
            self.paths = [Path(str(path).replace(str(find), str(replace))) for path in self.paths]
        else:
            raise TypeError("Unsupported type for find/replace.")


    def to_csv(self, file_name):
        """
        Saves the list of paths as a .csv file.

        Args:
            file_name (str): The name of the file where the CSV should be saved.
        """
        df = pd.DataFrame([str(path) for path in self.paths], columns=['path'])

        df.to_csv(file_name, index=False, header=False)


    def read_csv(self, file_name):
        """
        Reads the list of paths from a .csv file.

        Args:
            file_name (str): The name of the file where the paths are saved.

        Returns:
            PathList: A new PathList with the imported paths.
        """
        df = pd.read_csv(file_name, header=None)

        return PathList(df[0].tolist())


    def relative_to(self, path):
        """
        Converts the list of paths into the relative paths to another path identified by the passed arguments.

        Args:
            path (str or pathlib.Path): The path to compute relative paths to.

        Returns:
            PathList: A new PathList with the relative paths.
        """
        filtered_paths = PathList([Path(p.relative_to(path)) for p in self.paths])

        return filtered_paths


    def remove(self, paths=None):
        """
        Remove specified paths from the PathList.

        Args:
            paths (str, pathlib.Path, list, or None): The paths to remove. If None, no paths are removed.

        Returns:
            PathList: A new PathList with the specified paths removed.
        """
        if paths is None:
            return
        else:
            if not isinstance(paths, PathList):
                paths = PathList(paths)

        filtered_paths = PathList([p for p in self.paths if p not in paths.paths])

        return filtered_paths

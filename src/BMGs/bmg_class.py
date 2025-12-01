import re
from typing import Dict, Optional

import numpy as np
import pandas as pd

from config import CompositionColumns, DataPath


class BMGs():
    def __init__(self, s: Optional[np.ndarray] = None, properties: Dict = None, bmg_s: Optional[str] = None) -> None:
        """
        Initialize the BMGs class with a 56-dimensional vector 's' and a datapath to an Excel file.
        
        Parameters:
            s (list or array-like): vector
        """
        self.properties = properties
        if s is not None:
            if sum(s) < 1.1: s = np.array(s) * 100
            self.s = s
            self.bmg_s = self._generate_BMGs_string()
        elif bmg_s is not None:
            self.bmg_s = bmg_s
            self.s = self._generate_s_from_bmg_string()
        else:
            raise ValueError("Either 's' or 'bmg_s' must be provided.")
    
    def _generate_s_from_bmg_string(self):
        """
        Generate the vector 's' from the BMGs string.
        
        Returns:
            np.ndarray: The vector 's'
        """
        # Zr59.62Cu14.9Ni11.76Al7.84Ag5.88 根据这个字符串生成s
        s = np.zeros(len(CompositionColumns))
        
        # Use regular expressions to parse the BMGs string
        pattern = r"([A-Za-z]+)(\d+\.\d+|\d+)"
        matches = re.findall(pattern, self.bmg_s)
        
        # Create a dictionary from the matches
        bmg_dict = {element: float(value) for element, value in matches}
        
        # Map the dictionary to the s vector
        for element, value in bmg_dict.items():
            if element in CompositionColumns:
                index = CompositionColumns.index(element)
                s[index] = value
        
        return s
        
      
    def _generate_BMGs_string(self):
        """
        Generate a BMGs string based on the vector 's' and the columns of the Excel file.
        
        Returns:
            str: BMGs string
        """
        assert len(self.s) == len(CompositionColumns), "The length of the vector 's' must match the number of columns in the Excel file."

        # Create a dictionary from relevant columns and vector 's'

        col_s_dict = {col: self.s[i] for i, col in enumerate(CompositionColumns)}
        
        # Sort columns by the value in 's' and create the BMGs string
        sorted_cols = sorted(col_s_dict, key=lambda c: col_s_dict[c], reverse=True)
        bmg_string = ''.join([f"{col}{round(col_s_dict[col], 2)}" for col in sorted_cols if col_s_dict[col] != 0])
        
        return bmg_string
    
    def get_base_matrix(self):
        """
        Get the base matrix of the BMGs.
        
        Returns:
            str: base matrix
        """
        return CompositionColumns[np.argmax(self.s)]
    
    def __str__(self) -> str:
        return f'BMGs: {self.bmg_s}' + '\n' + f'Properties: {self.properties}'
    
    
def unit_test():
    s = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 15.4, 0.0, 0.0, 0.0, 23.1, 30.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 30.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    bmg = BMGs(s)
    print(bmg)
    print(bmg.get_base_matrix())
    bmg_s = "Zr59.62Cu14.9Ni11.76Al7.84Ag5.88"
    bmg = BMGs(bmg_s=bmg_s)
    print(bmg.s)
    print(bmg)
    print(bmg.get_base_matrix())

# python -m BMGs.bmg_class 
if __name__ == '__main__':
    unit_test()
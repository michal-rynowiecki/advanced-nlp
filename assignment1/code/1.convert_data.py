import random
from myutils import read_data, write_data

def transform_line(line: str):
    """
    Converts a string by introducing random noise. The noise is based on probability and can be
    removing the character, replacing it, or adding an additional character in front of it.
    
    Parameters
    ----------
    line: str
        The line to transform

    Returns
    -------
    new_line: str
        Transformed word with introduced noise
    """
    new_line = ""

    for i in range(len(line)):
        a = random.random()
        if a < 0.03:
            # Add an additional character
            new_line += line[i]*2
        elif a < 0.06:
            # Remove character
            continue
        elif a < 0.09:
            # Swap with previous
            if i != 0:
                new_line = new_line[:-1]
                new_line += line[i]
                new_line += line[i-1]
            # If the index is 0, continue
            else:
                new_line += line[i]
        else:
            # Do nothing
            new_line += line[i]

    return new_line

if __name__ == "__main__":
    
    random.seed(a=11)

    path = '/home/michal/Documents/advanced_nlp/assignment1/data/'
    in_file = 'sst.train'
    out_file = 'sst.transformed.train'

    txts, labels = read_data(path + in_file)

    transformed_txts = [transform_line(txt) for txt in txts]

    write_data(transformed_txts, labels, path+out_file)

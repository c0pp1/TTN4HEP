'''
Filename: /workspaces/metis-fpga/tutorial/higgs/ttn_format_converter.py
Path: /workspaces/metis-fpga/tutorial/higgs
Created Date: Tuesday, January 30th 2024, 5:24:43 pm
Author: c0pp1

Copyright (c) 2024 Tensor AI Solutions GmbH
'''

from dataclasses import dataclass
import subprocess
import numpy as np
import re  # regex
import glob

#@dataclass
class Index:
    """
    Index struct
    :param dim dimension of index
    :param id unique identifier
    :param tag name of index
    """
    dim: int
    id: int
    tag: str

    def __init__(self, dim_, id_, tag_):
        self.dim = dim_
        self.id = id_
        self.tag = tag_
        self.phys_id = None
        self.layer = None
        self.layer_to = None
        self.pos = None
        self.pos_to = None

        self.is_phys = 'p' in tag_
        self.is_virt = False
        self.is_label = 'l' in tag_

        if self.is_phys:
            self.phys_id = int(re.findall(r'\d+', tag_)[0])
        elif self.is_label:
            pass
        else: 
            self.is_virt = True
            target_split = tag_.split(',')
            self.layer = int(target_split[0].split('.')[0])
            self.layer_to = int(target_split[1].split('.')[0])
            self.pos = int(target_split[0].split('.')[1])
            self.pos_to = int(target_split[1].split('.')[1])
            if self.layer > self.layer_to:
                self.layer, self.layer_to = self.layer_to, self.layer
                self.pos, self.pos_to = self.pos_to, self.pos
                self.tag = f"{self.layer}.{self.pos},{self.layer_to}.{self.pos_to}"
            
    def __eq__(self, other):
        return self.dim == other.dim and self.id == other.id and self.tag == other.tag  

    def __gt__(self, other):

        if self.is_phys and other.is_phys:
            return self.phys_id > other.phys_id
        elif self.is_virt and other.is_virt:
            if self.layer == other.layer:
                if self.pos == other.pos:
                    if self.layer_to == other.layer_to:
                        return self.pos_to > other.pos_to
                    else:
                        print("Warning: comparing indices with different layer_to")
                        return self.layer_to > other.layer_to
                else:
                    print("Warning: comparing indices with different pos")
                    return self.pos > other.pos
            else:
                return self.layer < other.layer
        elif self.is_label and other.is_label:
            return False
        elif self.is_phys:
            return False
        elif self.is_virt:
            return not other.is_label
        elif self.is_label:
            return True
        else:
            raise ValueError("Error: comparing indices with different types")

    def __lt__(self, other):
        if self.is_phys and other.is_phys:
            return self.phys_id < other.phys_id
        elif self.is_virt and other.is_virt:
            if self.layer == other.layer:
                if self.pos == other.pos:
                    if self.layer_to == other.layer_to:
                        return self.pos_to < other.pos_to
                    else:
                        print("Warning: comparing indices with different layer_to")
                        return self.layer_to < other.layer_to
                else:
                    print("Warning: comparing indices with different pos")
                    return self.pos < other.pos
            else:
                return self.layer > other.layer
        elif self.is_label and other.is_label:
            return False
        elif self.is_phys:
            return True
        elif self.is_virt:
            return not other.is_phys
        elif self.is_label:
            return False
        else:
            raise ValueError("Error: comparing indices with different types")

    def __repr__(self):
        return f"Index(tag={self.tag}, phys_id={self.phys_id}, layer={self.layer}, layer_to={self.layer_to}, pos={self.pos}, pos_to={self.pos_to}, label={self.is_label})"



def parse_index(index_str: str) -> Index:
    """Helper function to parse index string of "ITensor-3" format.
    Indices are in the format: '(dim=4|id=1|"tag")'
    Args:
        index_str (str): string from parenthesis "(" to parenthesis ")"
    Returns:
        Index: Struct of index with dimension, id and tag
    """
    clean_str = index_str.strip("(").strip(")").replace('"', '').replace("id=", "").replace("dim=", "")
    idx = clean_str.split("|")
    return Index(int(idx[0]), int(idx[1]), str(idx[2]))


def parse_line(line_str: str):
    """
    "IndexVal: val = 3, ind = (dim=3|id=942|"label")  \
     IndexVal: val = 3, ind = (dim=3|id=34|"0.0,1.0") \
     IndexVal: val = 1, ind = (dim=3|id=306|"0.0,1.1")  element:-0.0040587"
    """
    # find Index values list, indexing from 0
    index_values = tuple([int(ii) -1 for ii in re.findall(r'val = (\d+)', line_str)])
    # find relative indices
    indexes = [parse_index(idx) for idx in re.findall(r'\((.*?)\)', line_str)]
    # find tensor value
    value = float(line_str.split("element:")[1].strip())
    
    return index_values, indexes, value
    
     
def parse_from_phoenix(filedir: str, filename: str = '', binary: bool = True, idtype=np.float64):
    """ Parse TTN text file in Phoenix format to list of Tensor structs
    :param filedir
    :param filename
    :param dtype: data type of tensor elements
    :return: list of Tensor structs
    """
    if filename == '':
        filename = "tensors*"
    else:
        filename = filename+"*" if not filename.endswith(".ttn") else filename[:-4]+"*"
    files = glob.glob(filename if not filename.endswith(".ttn") else filename[:-4], root_dir=filedir)

    # check if one file in files is .ttn and optionally .binary
    file = next((f for f in files if f.endswith(".ttn")), None)
    if file is None:
        raise FileNotFoundError(f"No .ttn file found in directory '{filedir}'")

    with open(filedir + file, 'r') as f:
        lines = f.readlines()
    if binary:
        # check if binary file exists
        if not next((f for f in files if f.endswith(".binary")), None):
            raise FileNotFoundError(f"No .binary file found in directory '{filedir}'")
        # read elements from binary file
        with open(filedir + file.replace(".ttn", ".binary"), 'rb') as f:
            elements = np.fromfile(f, dtype=idtype)
        
    if len(lines) != elements.size:
        raise TypeError("number of elements read in binary file does not match the expected one. Check the input data type")
    
    centipede = []
    for i, line in enumerate(lines):
        if not binary:
            centipede.append(parse_line(line))
        else:
            index_values, indexes, _ = parse_line(line)
            value = elements[i]
            centipede.append((index_values, indexes, value))

    ttn = []
    tt_legs = [centipede[0][1]]  # get first legs
    tens = None
    cc = 0
    for elem in centipede:
        i_val = elem[0]
        legs = elem[1]
        dims = [ll.dim for ll in legs]
        # check if legs are still the same (if its same )
        if not all([legs[ii] == tt_legs[cc][ii] for ii in range(len(legs))]):
            
            tt_legs.append(legs)
            cc += 1

        if tens is None:
            tens = np.zeros(dims, dtype=np.float64)

        tens[i_val] = elem[2]
        if i_val == tuple([ll.dim - 1 for ll in legs]):
            temp = [(l, i) for i, l in enumerate(legs)]
            temp.sort(key=lambda x: x[0])
            #print(*temp, tt_legs[-1], '\n', sep='\n')
            tt_legs[-1], permutation = zip(*temp)
            ttn.append(np.transpose(tens, permutation))
            tens = None

    # recast with Tensor struct
    tt_struct = []
    for ii in range(len(ttn)):
        tt_struct.append({'legs': tt_legs[ii], 'elements': ttn[ii], 'id_': ii+1})
    return tt_struct

def parse_from_metis(filedir: str, filename: str = '', binary: bool = True, idtype=np.float64):
    """ Parse TTN text file in Phoenix format to list of Tensor structs
    :param filedir
    :param filename
    :param dtype: data type of tensor elements
    :return: list of Tensor structs
    """
    if filename == '':
        filename = "*"
    else:
        filename = filename+"*" 
    files = glob.glob(filename, root_dir=filedir)

    modes_file = next((f for f in files if f.endswith("modes.txt") ), None)
    if modes_file is None:
        raise FileNotFoundError(f"No modes.txt file found in directory '{filedir}'")
    elements_file = next((f for f in files if f.endswith("elements.bin")), None)
    if elements_file is None:
        raise FileNotFoundError(f"No elements.bin file found in directory '{filedir}'")
    
    with open(filedir + modes_file, 'r') as file:
        lines = file.readlines()
    with open(filedir + elements_file, 'rb') as file:
        n_elements = np.fromfile(file, dtype=np.int32, count=1)[0]
        elements = np.fromfile(file, dtype=idtype)

    if n_elements != len(elements):
        raise TypeError(f"number of elements read in binary file ({len(elements)}) does not match the expected one ({n_elements}). Check the input data type")
    
    parsed = []
    legs = []
    dims = []
    for line in lines[2:]:  # skip first two lines
        if line.startswith("\n"):
            # elements are in column major order, thus revese dims
            dims.reverse()
            parsed.append({'legs': legs, 'elements': elements[:np.prod(dims)].reshape(dims).transpose([i for i in range(len(dims)-1, -1, -1)]), 'id_': len(parsed)})
            elements = elements[np.prod(dims):]
            legs = []
            dims = []
        else:
            tag, dim = line.split()
            dims.append(int(dim))
            if tag.startswith("v"):
                tag = tag[1:]
            legs.append(Index(int(dim), len(legs)+3*len(parsed), tag))
            if tag.startswith("p"):
                legs[-1].is_phys = True
            if tag.startswith("l"):
                legs[-1].is_label = True
    if len(legs) > 0:  
        dims.reverse()        
        parsed.append({'legs': legs, 'elements': elements[:np.prod(dims)].reshape(dims).transpose([i for i in range(len(dims)-1, -1, -1)]), 'id_': len(parsed)})
    return parsed

def parse_from_tenet(filedir: str, filename: str = '', binary: bool = True, idtype=np.float64):
    """ Parse TTN text file in Phoenix format to list of Tensor structs
    :param filedir
    :param filename
    :param dtype: data type of tensor elements
    :return: list of Tensor structs
    """
    if filename == '':
        filename = "*.npz"
    else:
        filename = filename+"*" if not filename.endswith(".npz") else filename[:-4]+"*"
    files = glob.glob(filename, root_dir=filedir)
    file = next((f for f in files if f.endswith(".npz")), None)
    if file is None:
        raise FileNotFoundError(f"No .npz file found in directory '{filedir}'")
    tensors = np.load(filedir + file)
    
    parsed = []
    n_layers = sorted([int(key.split('.')[0]) for key in tensors.keys()])[-1]+1
    for key in tensors.keys():
        if key == "0.0":
            tags = ["0.0,1.0", "0.0,1.1", "label"]
        elif int(key.split('.')[0]) == n_layers-1:
            tags = [f"p{2**int(key.split('.')[1])}", f"p{2**int(key.split('.')[1])+1}", f"{int(key.split('.')[0])-1}.{int(key.split('.')[1])//2},"+key]
        else:
            tags = [f"{key},{key.split('.')[0]+1}.{2**int(key.split('.')[1])}", 
                    f"{key},{key.split('.')[0]+1}.{2**int(key.split('.')[1])+1}", 
                    f"{int(key.split('.')[0])-1}.{int(key.split('.')[1])//2},"+key]
        parsed.append({'legs': [Index(tensors[key].shape[i], i+3*len(parsed), tags[i]) for i in range(len(tensors[key].shape))], 'elements': tensors[key], 'id_': len(parsed)})
    return parsed

def to_metis(parsed, folder_name: str, filename: str = 'tensors', odtype=np.float64):
    subprocess.call(["mkdir", "-p", folder_name])
    modes = open(folder_name + (filename + '_' if filename else '') + "modes.txt", 'w')
    modes.write("# TTN modes\n")
    for tensor in parsed:
        modes.write("\n")
        for leg in tensor['legs']:
            tag = ""
            if leg.is_phys:
                tag = "p"+str(leg.phys_id)
            else:
                if leg.is_label:
                    tag = leg.tag
                else:
                    tag = "v"+f"{leg.layer}.{leg.pos},{leg.layer_to}.{leg.pos_to}"
            modes.write(f"{tag} {leg.dim}\n")
        
    modes.close()

    elements = folder_name + (filename + '_' if filename else '') + "elements.bin"

    with open(elements, 'wb') as elements:
        elements.write(np.array([tensor['elements'].size for tensor in parsed]).sum(dtype=np.int32).tobytes('F'))
        for tensor in parsed:
            elements.write(tensor['elements'].astype(odtype).tobytes('F'))


def to_phoenix(parsed, folder_name: str, filename: str = 'tensors', odtype=np.float64):
    subprocess.call(["mkdir", "-p", folder_name])
    with open(folder_name + (filename if filename else "tensors") + ".ttn", 'w') as file:
        for tensor in parsed:
            for idx3 in range(tensor['legs'][2].dim):
                for idx2 in range(tensor['legs'][1].dim):
                    for idx1 in range(tensor['legs'][0].dim):
                        file.write(f"IndexVal: val = {idx1+1}, ind = (dim={tensor['legs'][0].dim}|id={tensor['legs'][0].id}|\"{tensor['legs'][0].tag}\")")
                        file.write(f"IndexVal: val = {idx2+1}, ind = (dim={tensor['legs'][1].dim}|id={tensor['legs'][1].id}|\"{tensor['legs'][1].tag}\")")
                        file.write(f"IndexVal: val = {idx3+1}, ind = (dim={tensor['legs'][2].dim}|id={tensor['legs'][2].id}|\"{tensor['legs'][2].tag}\")")
                        file.write(f"\telement:{tensor['elements'][idx1, idx2, idx3]}\n")

    
    with open(folder_name + (filename if filename else "tensors") + ".binary", 'wb') as file:
        for tensor in parsed:
            file.write(tensor['elements'].astype(odtype).tobytes('F'))


def to_tenet(parsed, folder_name: str, filename: str = 'tensors', odtype=np.float64):
    subprocess.call(["mkdir", "-p", folder_name])
    np.savez(folder_name + (filename if filename else "tensors"), **{"0.0" if tensor['legs'][-1].is_label else f"{tensor['legs'][-1].layer_to}.{tensor['legs'][-1].pos_to}": tensor['elements'].astype(odtype) for tensor in parsed})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Convert TTN text file to folder format')
    parser.add_argument('-i', '--input', type=str, help='input folder')
    parser.add_argument('-f', '--filename', type=str, help='input filename', default='')
    parser.add_argument('-if', '--input_format', type=str, help='input format', default='phoenix', choices=['phoenix', 'metis', 'tenet'])
    parser.add_argument('-o', '--output', type=str, help='output folder')
    parser.add_argument('-of', '--output_format', type=str, help='output format', default='tenet', choices=['phoenix', 'metis', 'tenet'])
    parser.add_argument('-ot', '--odtype', type=str, help='output data type', default='float64')
    parser.add_argument('-b', '--binary', type=bool, help='wether or not to read elements from binary file', default=True)
    parser.add_argument('-it', '--idtype', type=str, help='data type for elements in input binary file', default='float64')
    args = parser.parse_args()
    exec("file_parser = parse_from_" + args.input_format)
    exec("file_writer = to_" + args.output_format)

    parsed = file_parser(args.input, args.filename, args.binary, args.idtype)
    file_writer(parsed, args.output, args.filename, args.odtype)

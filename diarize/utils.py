import os
import sys
from collections import namedtuple


class Dict2ObjParser:
    # Original code from https://stackoverflow.com/questions/6866600/how-to-parse-read-a-yaml-file-into-a-python-object
    def __init__(self, nested_dict):
        self.nested_dict = nested_dict

    def parse(self):
        nested_dict = self.nested_dict
        if (obj_type := type(nested_dict)) is not dict:
            raise TypeError(f"Expected 'dict' but found '{obj_type}'")
        return self._transform_to_named_tuples("root", nested_dict)

    def _transform_to_named_tuples(self, tuple_name, possibly_nested_obj):
        if type(possibly_nested_obj) is dict:
            named_tuple_def = namedtuple(tuple_name,
                                         possibly_nested_obj.keys())
            transformed_value = named_tuple_def(*[
                self._transform_to_named_tuples(key, value)
                for key, value in possibly_nested_obj.items()
            ])
        elif type(possibly_nested_obj) is list:
            transformed_value = [
                self._transform_to_named_tuples(f"{tuple_name}_{i}",
                                                possibly_nested_obj[i])
                for i in range(len(possibly_nested_obj))
            ]
        else:
            transformed_value = possibly_nested_obj

        return transformed_value


def read_inputlist(input_list):
    # Read the input file which contains the paths to wavfiles
    if os.path.isfile(input_list) == False:
        print("No such file : ", input_list)
        sys.exit(1)

    wav_list = []
    with open(input_list, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if os.path.isfile(line) == False:
                print("No such wav file : ", line)
                continue
            else:
                wav_list.append(line)

    print("# of wav files : ", len(wav_list))

    return wav_list


def write_rttm(SEC_tuples, out_rttm_file):
    # Write the rttm file given SEC_tuples (start, end, cluster_id)
    file_id = out_rttm_file.split('/')[-1].replace('.rttm', '')
    place_holder = [
        'SPEAKER', file_id, '1', '0', '0', '<NA>', '<NA>', '0', '<NA>', '<NA>'
    ]

    with open(out_rttm_file, 'w') as f_output:
        for tup in SEC_tuples:
            place_holder[3] = "%.3f" % tup[0]
            place_holder[4] = "%.3f" % (tup[1] - tup[0])
            place_holder[7] = str(tup[2])

            output_string = ' '.join(place_holder)

            f_output.write(output_string + '\n')


def read_vadfile(vad_file):
    # Parse the vadfile which contain
    vad_segments = []

    with open(vad_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            start, end = line.split(' ')[0], line.split(' ')[1]
            vad_segments.append((float(start), float(end)))

    return vad_segments

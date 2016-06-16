'''
Created on Jul 16, 2014

@author: Philip Schulz

I use the original Brown et al. channel terminology here when talking about source and target.
'''

import argparse, sys

def __sort_alignment(alignment):
    return sorted(list(alignment), key=lambda entry : (entry[0], entry[1]))

def mosesWrapper(heuristic):
    def readFiles(src2tgtFile, tgt2srcFile, extension, file_name):
        extension = "-".join(extension.split("_"))
        with open(src2tgtFile) as src2tgtStream, open(tgt2srcFile) as tgt2srcStream, open(file_name + "." + extension, "w") as out:
            i = 0
            for src2tgtLinks in src2tgtStream:
                i = i + 1
                sys.stderr.write("reading line " + str(i) + "\r")
                src2tgtAlignment = set([(int(link[0]), int(link[1])) for link in map(lambda x : x.split('-'), src2tgtLinks.split())])
                tgt2srcAlignment = set([(int(link[1]), int(link[0])) for link in map(lambda x : x.split('-'), next(tgt2srcStream).split())]) 
                merged_alignment = heuristic(src2tgtAlignment, tgt2srcAlignment)
                for link in merged_alignment:
                    out.write(str(link[0]) + "-" + str(link[1]) + " ")
                # end line
                out.write("\n")
                
    return readFiles

def naaclWrapper(heuristic):
    def readFiles(src2tgtFile, tgt2scrFile, extension, file_name):
        extension = "-".join(extension.split("_"))
        with open(src2tgtFile) as src2tgtStream, open(tgt2scrFile) as tgt2scrStream, open(file_name + "." + extension, "w") as out:
            sentenceNum = 1
            src2tgtAlignment = set()
            tgt2scrAlignment = set()
            
            for line in src2tgtStream:
                sys.stderr.write("processing sentence " + str(sentenceNum) + "\r")
                sys.stderr.flush()
                src2tgtFields = [int(x) for x in line.split(' ')]
                if src2tgtFields[0] == sentenceNum:
                    src2tgtAlignment.add(tuple(src2tgtFields[1:]))
                else:
                    for tgt2srcLinks in tgt2scrStream:
                        tgt2srcFields = [int(x) for x in tgt2srcLinks.split(' ')]
                        if tgt2srcFields[0] != sentenceNum:
                            tgt2scrAlignment.add(tuple(tgt2srcFields[1:]))
                        else:
                            # write current sentence alignment
                            merged_alignment = heuristic(src2tgtAlignment, tgt2scrAlignment)
                            for link in merged_alignment:
                                out.write(str(sentenceNum) + " " + str(link[0]) + " " + str(link[1]))
                                out.write("\n")
                        
                            src2tgtAlignment = set()
                            src2tgtAlignment.add(tuple(src2tgtFields[1:]))
                            tgt2scrAlignment = set()
                            tgt2scrAlignment.add(tuple(tgt2srcFields[1:]))
                            sentenceNum += 1
                
    return readFiles
                    
@naaclWrapper
def naacl_union(src2tgtLinks, tgt2srcLinks):
    return __sort_alignment(__union(src2tgtLinks, tgt2srcLinks)[0])

@naaclWrapper
def naacl_intersection(src2tgtLinks, tgt2srcLinks):
    return __sort_alignment(__intersection(src2tgtLinks, tgt2srcLinks)) 
             
@naaclWrapper
def naacl_grow(src2tgtLinks, tgt2srcLinks):
    return __sort_alignment(__grow(src2tgtLinks, tgt2srcLinks))

@naaclWrapper
def naacl_grow_diag(src2tgtLinks, tgt2srcLinks):
    return __sort_alignment(__grow(src2tgtLinks, tgt2srcLinks, True))

@naaclWrapper
def naacl_grow_diag_final(src2tgtLinks, tgt2srcLinks):
    return __sort_alignment(__grow(src2tgtLinks, tgt2srcLinks, True, True))

@naaclWrapper
def naacl_grow_diag_final_and(src2tgtLinks, tgt2srcLinks):
    return __sort_alignment(__grow(src2tgtLinks, tgt2srcLinks, True, True, True))
                                 
@mosesWrapper
def moses_union(src2tgtLinks, tgt2srcLinks):
    return __sort_alignment(__union(src2tgtLinks, tgt2srcLinks)[0])
    
def __union(src2tgtLinks, tgt2srcLinks):
    
    union_alignment = src2tgtLinks.union(tgt2srcLinks)
        
    return union_alignment, src2tgtLinks, tgt2srcLinks

@mosesWrapper
def moses_intersection(src2tgtLinks, tgt2srcLinks):
    return __sort_alignment(__intersection(src2tgtLinks, tgt2srcLinks))

def __intersection(src2tgtLinks, tgt2srcLinks):
        
    intersection_alignment = src2tgtLinks & tgt2srcLinks
    
    return intersection_alignment
    
@mosesWrapper 
def moses_grow_diag(src2tgtLinks, tgt2srcLinks):
    return __sort_alignment(__grow(src2tgtLinks, tgt2srcLinks, True))

@mosesWrapper
def moses_grow(src2tgtLinks, tgt2srcLinks):
    return __sort_alignment(__grow(src2tgtLinks, tgt2srcLinks))

@mosesWrapper
def moses_grow_diag_final(src2tgtLinks, tgt2srcLinks):
    return __sort_alignment(__grow(src2tgtLinks, tgt2srcLinks, True, True))

@mosesWrapper
def moses_grow_diag_final_and(src2tgtLinks, tgt2srcLinks):
    return __sort_alignment(__grow(src2tgtLinks, tgt2srcLinks, True, True, True))
    
def __grow(src2tgtLinks, tgt2srcLinks, diag=False, final=False, final_and=False):
    intersected_alignment = __intersection(src2tgtLinks, tgt2srcLinks)
    union_alignment, src2tgtLinks, tgt2srcLinks = __union(src2tgtLinks, tgt2srcLinks)
    
    src_candidates = list(set(link[0] for link in union_alignment))
    src_candidates.sort()
    tgt_candidates = list(set(link[1] for link in union_alignment))
    tgt_candidates.sort()
    
    src_aligned = set(link[0] for link in intersected_alignment)
    tgt_aligned = set(link[1] for link in intersected_alignment)
    
    grownAlignment = set(intersected_alignment)
    
    while True:
        added = False
        # need to iterate first over target and then over source language
        # => this is wrongly described in Moses manual
        for tgt_pos in tgt_candidates:
            for src_pos in src_candidates:
                if (src_pos, tgt_pos) in grownAlignment:
                    
                    for step in [-1, 1]:
                        new_src_pos = src_pos + step
                        if new_src_pos not in src_aligned:
                            new_link = (new_src_pos, tgt_pos)
                            if new_link in union_alignment:
                                grownAlignment.add(new_link)
                                src_aligned.add(new_src_pos)
                                added = True
                
                    for step in [-1, 1]:
                        new_tgt_pos = tgt_pos + step
                        if new_tgt_pos not in tgt_aligned:
                            new_link = (src_pos, new_tgt_pos)
                            if new_link in union_alignment:
                                grownAlignment.add(new_link)
                                tgt_aligned.add(new_tgt_pos)
                                added = True
                    
                    # extend alignment diagonally -> my version
                    if diag:
                        for src_step in [-1, 1]:
                            for tgt_step in [-1, 1]:
                                new_src_pos = src_pos + src_step
                                new_tgt_pos = tgt_pos + tgt_step
                                if new_src_pos not in src_aligned or new_tgt_pos not in tgt_aligned:
                                    new_link = (new_src_pos, new_tgt_pos)
                                    if new_link in union_alignment:
                                        grownAlignment.add(new_link)
                                        src_aligned.add(new_src_pos)
                                        tgt_aligned.add(new_tgt_pos)
                                        added = True    
        if not added:
            break
                                    
    if final and not final_and:                                
        for tgt_pos in tgt_candidates:
            for src_pos in src_candidates:
                if src_pos not in src_aligned or tgt_pos not in tgt_aligned:
                    new_link = (src_pos, tgt_pos)
                    if new_link in src2tgtLinks:
                        grownAlignment.add(new_link)
                        src_aligned.add(src_pos)
                        tgt_aligned.add(tgt_pos)
                        
        
        for tgt_pos in tgt_candidates:
            for src_pos in src_candidates:
                if src_pos not in src_aligned or tgt_pos not in tgt_aligned:
                    new_link = (src_pos, tgt_pos)
                    if new_link in tgt2srcLinks:
                        grownAlignment.add(new_link)
                        src_aligned.add(src_pos)
                        tgt_aligned.add(tgt_pos)

    if final_and:
        for tgt_pos in tgt_candidates:
            for src_pos in src_candidates:
                if src_pos not in src_aligned and tgt_pos not in tgt_aligned:
                    new_link = (src_pos, tgt_pos)
                    if new_link in src2tgtLinks:
                        grownAlignment.add(new_link)
                        src_aligned.add(src_pos)
                        tgt_aligned.add(tgt_pos)
                    
    
        for tgt_pos in tgt_candidates:
            for src_pos in src_candidates:
                if src_pos not in src_aligned and tgt_pos not in tgt_aligned:
                    new_link = (src_pos, tgt_pos)
                    if new_link in tgt2srcLinks:
                        grownAlignment.add(new_link)
                        src_aligned.add(src_pos)
                        tgt_aligned.add(tgt_pos)
    
    return grownAlignment

def main():
    # create commandLinksParser
    commandLineParser = argparse.ArgumentParser("Merges the alignment files produced by aligning "
    "a bilingual corpus in both directions. Different heuristics for the merger must be specified. "
    "Note that the different mergers are mutually exclusive, meaning that only one of them can and "
    "must be specified. The files are merged according to the order in the first file. Usually, this "
    "file should have entries of the form 'target-source' (in IBM terminolgy -- in Moses terminology this "
    "would be 'source-target'). This is the format that Moses needs for "
    "phrase extraction. Hence the second file should have entries of the form 'source-target' (or 'target-source "
    "in Moses terms).")
    
    commandLineParser.add_argument("alignmentFiles", nargs=2, help="The two " 
    "alignment files that have been generated from aligning the corpus in different directions.")
    
    commandLineParser.add_argument("--filename", default = "aligned", help="Set the file_name of the output file. "
                                   "An extension correpsonding to the chosen symmetrization heuristic will be added.")
    
    commandLineParser.add_argument("--format", default = "moses", nargs = 1, help = "Choose the input and output format of the alignment files. "
                                   "By default, moses format will be assumed. The alternative is 'naacl'.")
    
    # add group of alignment heuristics that are mutually exclusive but one of them is required
    heuristics = commandLineParser.add_mutually_exclusive_group(required=True)
    
    # create intersection argument
    heuristics.add_argument("--intersect", action="store_true", help="Triggers the "
    "intersection heuristic for merging. Only alignment points that are present in BOTH alignments "
    "are retained. This lead to high-precision, low-recall alignments. Due to the rather unrestrictive "
    "nature of the resulting alignment, subsequent steps in the training pipeline will extract a very "
    "large amount of phrases.")
    
    # create union argument
    heuristics.add_argument("--union", action="store_true", help="Triggers the union "
    "heuristic for merging. All points that occur in EITHER alignment are retained. This leads to "
    "high-recall alignments.")
    
    # create grow argument
    heuristics.add_argument("--grow", action="store_true", help="Triggers the grow heuristic for merging. "
                            "Expands the intersected alignment in horizontal and vertical direction until no "
                            "new points can be added.")
    
    # create grow-diag argument
    heuristics.add_argument("--grow-diag", action="store_true", help="Triggers the "
    "grow-diag heuristic for merging. This merger first executes the intersection merger and then "
    "adds any alignment points that are a) neighbouring a retained alignment point and b) present in "
    "EITHER of the original alignments.") 
    
    # create grow-diag-final argument
    heuristics.add_argument("--grow-diag-final", action="store_true", help="Triggers the "
    "grow-diag-final heuristic for merging. This merger executes the grow-diag merger and then adds "
    "alignment points for unaligned word on EITHER side, given that they are aligned in EITHER alignment."
    "This may effectively introduce 'isolated' alignment points that are not in the vicinity of any "
    "other alignment points.")
    
    # create grow-diag-final-and argument
    heuristics.add_argument("--grow-diag-final-and", action="store_true", help="Triggers "
    "the grow-diag-final-and heuristc for merging. This merger is just like the grow-diag-final merger "
    "with the stronger requirement that yet unaligned words may only be linked to other yet unaligned words.")

    args = vars(commandLineParser.parse_args())
    src2tgtFile = args["alignmentFiles"][0]
    tgt2srcFile = args["alignmentFiles"][1]
    file_name = args["filename"]
    fileFormat = args["format"][0].lower()
    
    if args["union"]:
        if fileFormat == "moses":
            moses_union(src2tgtFile, tgt2srcFile, "union", file_name)
        elif fileFormat == "naacl":
            naacl_union(src2tgtFile, tgt2srcFile, "union", file_name)
    elif args["intersect"]:
        if fileFormat == "moses":
            moses_intersection(src2tgtFile, tgt2srcFile, "intersect", file_name)
        elif fileFormat == "naacl":
            naacl_intersection(src2tgtFile, tgt2srcFile, "intersect", file_name)
    elif args["grow"]:
        if fileFormat == "moses":
            moses_grow(src2tgtFile, tgt2srcFile, "grow", file_name)
        elif fileFormat == "naacl":
            naacl_grow(src2tgtFile, tgt2srcFile, "grow", file_name)
    elif args["grow_diag"]:
        if fileFormat == "moses":
            naacl_grow_diag(src2tgtFile, tgt2srcFile, "grow-diag", file_name)
        elif fileFormat == "naacl":
            naacl_grow_diag(src2tgtFile, tgt2srcFile, "grow-diag", file_name)
    elif args["grow_diag_final"]:
        if fileFormat == "moses":
            moses_grow_diag_final(src2tgtFile, tgt2srcFile, "grow-diag-final", file_name)
        elif fileFormat == "naacl":
            naacl_grow_diag_final(src2tgtFile, tgt2srcFile, "grow-diag-final", file_name)
    elif args["grow_diag_final_and"]:
        if fileFormat == "moses":
            moses_grow_diag_final_and(src2tgtFile, tgt2srcFile, "grow-diag-final-and", file_name)
        elif fileFormat == "naacl":
            naacl_grow_diag_final_and(src2tgtFile, tgt2srcFile, "grow_diag-final-and", file_name)

if __name__ == '__main__':
    main()

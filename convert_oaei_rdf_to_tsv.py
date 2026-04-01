'''
Convert OAEI RDF/XML alignment files to DeepOnto-compatible TSV format

The OAEI alignment format uses an RDF/XML schema with <Cell> elements containing:
    (fields)     <entity1>, <entity2>, <relation>, and <measure>    

This script extracts equivalence mappings and writes them as TSV files with the header:
    SrcEntity \t  TgtEntity \t   Score

Usage:
    # converts a single file
    python convert_oaei_rdf_to_tsv.py reference/cmt-conference.rdf
    # convert all .rdf files in a directory
    python convert_oaei_rdf_to_tsv.py reference/
    # specify output directory
    python convert_oaei_rdf_to_tsv.py reference/ --output-dir refs_equiv/
    # library usage
    from convert_oaei_rdf_to_tsv import convert_rdf_alignment
    convert_rdf_alignment("cmt-conference.rdf", "cmt-conference.tsv")
'''

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# known OAEI alignment namespace URIs across different editions
# the default (unprefixed) namespace in the XML determines which one is active
KNOWN_ALIGNMENT_NS = [
    "http://knowledgeweb.semanticweb.org/heterogeneity/alignment",
    "http://knowledgeweb.semanticweb.org/heterogeneity/alignment#",
    "http://knowledgeweb.semantics.org/ontoalign/align#",
    "http://knowledgeweb.semantics.org/ontoalign/align",
]

RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"


def _detect_default_namespace(root):
    """
    detect the default namespace from the root element's tag
    ElementTree expands '{namespace}localname' for elements in a namespace
    If the root tag starts with '{', extract the namespace URI; 
    also scans child elements to find the alignment namespace.
    """
    tag = root.tag
    # checks root tag
    if tag.startswith("{"):
        ns = tag[1:tag.index("}")]
        return ns

    # if root is rdf:RDF, scan children for alignment NS
    for child in root.iter():
        ctag = child.tag
        if ctag.startswith("{"):
            ns = ctag[1:ctag.index("}")]
            if ns != RDF_NS:  # skip the RDF namespace itself
                return ns

    return None


def _find_elements(root, local_name, default_ns=None):
    """
    Find elements by local name, using the detected default namespace. Tries: 
    (1) the detected default namespace
    (2) all known OAEI namespaces
    (3) bare unnamespaced lookup
    """
    # build list of namespace URIs to try, most likely first
    ns_candidates = []
    if default_ns:
        ns_candidates.append(default_ns)
    ns_candidates.extend(KNOWN_ALIGNMENT_NS)

    for ns in ns_candidates:
        found = root.findall(f".//{{{ns}}}{local_name}")
        if found:
            return found

    # bare name fallback (no namespace)
    found = root.findall(f".//{local_name}")
    return found


def _get_child_text(parent, local_name, default_ns=None):
    """get text content of a child element by local name"""
    ns_candidates = []
    if default_ns:
        ns_candidates.append(default_ns)
    ns_candidates.extend(KNOWN_ALIGNMENT_NS)

    for ns in ns_candidates:
        elem = parent.find(f"{{{ns}}}{local_name}")
        if elem is not None and elem.text:
            return elem.text.strip()

    # bare name fallback
    elem = parent.find(local_name)
    if elem is not None and elem.text:
        return elem.text.strip()

    return None


def _get_child_resource(parent, local_name, default_ns=None):
    """Get rdf:resource URI from a child element"""
    ns_candidates = []
    if default_ns:
        ns_candidates.append(default_ns)
    ns_candidates.extend(KNOWN_ALIGNMENT_NS)

    for ns in ns_candidates:
        elem = parent.find(f"{{{ns}}}{local_name}")
        if elem is not None:
            # try rdf:resource attribute
            uri = elem.get(f"{{{RDF_NS}}}resource")
            if uri:
                return uri.strip()
            # try bare resource attribute
            uri = elem.get("resource")
            if uri:
                return uri.strip()

    # bare name fallback
    elem = parent.find(local_name)
    if elem is not None:
        uri = elem.get(f"{{{RDF_NS}}}resource")
        if uri:
            return uri.strip()
        uri = elem.get("resource")
        if uri:
            return uri.strip()

    return None



def parse_rdf_alignment(rdf_path):
    """
    parses an OAEI RDF/XML alignment file

    Parameters
    ----------
    rdf_path : str or Path
        Path to the RDF/XML alignment file.

    Returns
    -------
    list of dict
        Each dict has keys: 'entity1', 'entity2', 'relation', 'measure'.
        'relation' is typically '=' for equivalence.
        'measure' is a float confidence score (default 1.0).
    """
    rdf_path = Path(rdf_path)
    tree = ET.parse(rdf_path)
    root = tree.getroot()

    # auto-detect the default namespace from the XML structure
    default_ns = _detect_default_namespace(root)

    cells = _find_elements(root, "Cell", default_ns)
    mappings = []

    for cell in cells:
        entity1 = _get_child_resource(cell, "entity1", default_ns)
        entity2 = _get_child_resource(cell, "entity2", default_ns)
        relation = _get_child_text(cell, "relation", default_ns) or "="
        measure_str = _get_child_text(cell, "measure", default_ns)

        try:
            measure = float(measure_str) if measure_str else 1.0
        except ValueError:
            measure = 1.0

        if entity1 and entity2:
            mappings.append({
                "entity1": entity1,
                "entity2": entity2,
                "relation": relation,
                "measure": measure,
            })

    return mappings


def convert_rdf_alignment(rdf_path, tsv_path, relation_filter="="):
    """
    converts an OAEI RDF/XML alignment file to a DeepOnto-compatible TSV

    Parameters
    ----------
    rdf_path : str or Path
        Path to the input RDF/XML alignment file.
    tsv_path : str or Path
        Path to the output TSV file.
    relation_filter : str or None
        If set, only include mappings with this relation type.
        Use '=' for equivalence only, None for all relations.

    Returns
    -------
    int
        Number of mappings written.
    """
    rdf_path = Path(rdf_path)
    tsv_path = Path(tsv_path)

    mappings = parse_rdf_alignment(rdf_path)

    if relation_filter:
        mappings = [m for m in mappings if m["relation"] == relation_filter]

    tsv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(tsv_path, "w") as f:
        f.write("SrcEntity\tTgtEntity\tScore\n")
        for m in mappings:
            f.write(f"{m['entity1']}\t{m['entity2']}\t{m['measure']}\n")

    return len(mappings)


def convert_directory(input_dir, output_dir=None, relation_filter="="):
    """
    convert all .rdf alignment files in a directory to TSV

    Parameters
    ----------
    input_dir : str or Path
        Directory containing .rdf files.
    output_dir : str or Path or None
        Output directory for TSV files.  If None, creates a 'refs_equiv'
        sibling directory alongside the input directory.
    relation_filter : str or None
        Passed to convert_rdf_alignment().

    Returns
    -------
    list of tuple
        (rdf_filename, tsv_filename, mapping_count) for each converted file.
    """
    input_dir = Path(input_dir)

    if output_dir is None:
        output_dir = input_dir.parent / "refs_equiv"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    rdf_files = sorted(input_dir.glob("*.rdf"))

    if not rdf_files:
        print(f"WARNING: No .rdf files found in {input_dir}", file=sys.stderr)
        return results

    for rdf_file in rdf_files:
        tsv_name = rdf_file.stem + ".tsv"
        tsv_path = output_dir / tsv_name

        count = convert_rdf_alignment(rdf_file, tsv_path, relation_filter)
        results.append((rdf_file.name, tsv_name, count))
        print(f"  {rdf_file.name} -> {tsv_name}  ({count} mappings)")

    return results



def main():
    parser = argparse.ArgumentParser(
        description="Convert OAEI RDF/XML alignment files to DeepOnto-compatible TSV."
    )
    parser.add_argument(
        "input",
        help="Path to a single .rdf file or a directory of .rdf files.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Output directory for TSV files. Default: refs_equiv/ sibling of input.",
    )
    parser.add_argument(
        "--all-relations",
        action="store_true",
        default=False,
        help="Include all relation types, not just equivalence (=).",
    )

    args = parser.parse_args()
    input_path = Path(args.input)
    relation_filter = None if args.all_relations else "="

    if input_path.is_file():
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            tsv_path = output_dir / (input_path.stem + ".tsv")
        else:
            tsv_path = input_path.with_suffix(".tsv")

        count = convert_rdf_alignment(input_path, tsv_path, relation_filter)
        print(f"Converted {input_path.name} -> {tsv_path.name}  ({count} mappings)")

    elif input_path.is_dir():
        results = convert_directory(input_path, args.output_dir, relation_filter)
        total = sum(r[2] for r in results)
        print(f"\nConverted {len(results)} files, {total} total mappings.")

    else:
        print(f"ERROR: {input_path} is not a file or directory.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
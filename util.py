import xml.etree.ElementTree as eT
import pandas as pd


class XML2DataFrame:
    def __init__(self, xml_data):
        self.root = eT.parse(xml_data).getroot()

    def parse_root(self, root):
        """Return a list of dictionaries from the text and attributes of the children under this XML root."""
        return [self.parse_element(child) for child in root.getchildren()]

    def parse_element(self, element, parsed=None):
        """Collect {key:attribute} and {tag:text} from the XML element
            and all its children into a single dictionary of strings."""
        if parsed is None:
            parsed = dict()
        for key in element.keys():
            if key not in parsed:
                parsed[key] = element.attrib.get(key)
        if element.text:
            parsed[element.tag] = element.text
        for child in element.getchildren():
            self.parse_element(child, parsed)
        return parsed

    def process_data(self):
        """Initiate the root XML, parse it, and return a DataFrame."""
        structure_data = self.parse_root(self.root)
        return pd.DataFrame(structure_data)

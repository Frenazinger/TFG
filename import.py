from util import XML2DataFrame


def flows_to_dataframe(list_path):
    """Importar flujos en XML a DataFrame"""
    xml_flows = []
    for path in list_path:
        xml_flows.append(XML2DataFrame(path))
    xml_dataframe = None
    for xml_flow in xml_flows:
        if xml_dataframe is None:
            xml_dataframe = xml_flow.process_data()
        else:
            xml_dataframe = xml_dataframe.append(xml_flow.process_data(), sort=True)
    return xml_dataframe


# Importamos los XML. Eliminamos los atributos que ya no usaremos y crearemos typeAttack y date
# para poder hacer un posterior análisis a nivel de día o tipo de ataque.
# No hemos usado para identificar el día los campos startDateTime o stopDateTime
# ya que sus valores se solapan en diferentes días
# flows_of_20100612 = flows_to_dataframe(['data/labeled_flows_xml/TestbedSatJun12Flows.xml'])
# flows_of_20100612 = flows_of_20100612.drop(['TestbedSatJun12'], axis=1)
# flows_of_20100612['typeAttack'] = flows_of_20100612.Tag.replace(['Attack', 'Normal'], ['Brute Force SSH', 'N/A'])
# flows_of_20100612['date'] = '20100612'

flows_of_20100613 = flows_to_dataframe(['data/labeled_flows_xml/TestbedSunJun13Flows.xml'])
flows_of_20100613 = flows_of_20100613.drop(['TestbedSunJun13Flows'], axis=1)
flows_of_20100613['typeAttack'] = flows_of_20100613.Tag.replace(['Attack', 'Normal'], ['Infiltration', 'N/A'])
flows_of_20100613['date'] = '20100613'

flows_of_20100614 = flows_to_dataframe(['data/labeled_flows_xml/TestbedMonJun14Flows.xml'])
flows_of_20100614 = flows_of_20100614.drop(['TestbedMonJun14Flows'], axis=1)
flows_of_20100614['sourcePayloadAsUTF'] = 'nan'
flows_of_20100614['typeAttack'] = flows_of_20100614.Tag.replace(['Attack', 'Normal'], ['HTTP DoS', 'N/A'])

flows_of_20100614['date'] = '20100614'

flows_of_20100615 = flows_to_dataframe(['data/labeled_flows_xml/TestbedTueJun15-1Flows.xml',
                                        'data/labeled_flows_xml/TestbedTueJun15-2Flows.xml',
                                        'data/labeled_flows_xml/TestbedTueJun15-3Flows.xml'])
flows_of_20100615 = flows_of_20100615.drop(['TestbedTueJun15-1Flows',
                                            'TestbedTueJun15-2Flows',
                                            'TestbedTueJun15-3Flows',
                                            'sensorInterfaceId'], axis=1)
flows_of_20100615['typeAttack'] = flows_of_20100615.Tag.replace(['Attack', 'Normal'], ['Distributed DoS', 'N/A'])
flows_of_20100615['date'] = '20100615'

# flows_of_20100616 = flows_to_dataframe(['data/labeled_flows_xml/TestbedWedJun16-1Flows.xml',
#                                         'data/labeled_flows_xml/TestbedWedJun16-2Flows.xml',
#                                         'data/labeled_flows_xml/TestbedWedJun16-3Flows.xml'])
# flows_of_20100616 = flows_of_20100616.drop(['TestbedWedJun16-1Flows',
#                                             'TestbedWedJun16-2Flows',
#                                             'TestbedWedJun16-3Flows',
#                                             'startTime'], axis=1)
# flows_of_20100616['typeAttack'] = flows_of_20100616.Tag.replace(['Attack', 'Normal'], ['Brute Force SSH', 'N/A'])
# flows_of_20100616['date'] = '20100616'

flows_of_20100617 = flows_to_dataframe(['data/labeled_flows_xml/TestbedThuJun17-1Flows.xml',
                                        'data/labeled_flows_xml/TestbedThuJun17-2Flows.xml',
                                        'data/labeled_flows_xml/TestbedThuJun17-3Flows.xml'])
flows_of_20100617 = flows_of_20100617.drop(['TestbedThuJun17-1Flows',
                                            'TestbedThuJun17-2Flows',
                                            'TestbedThuJun17-3Flows'], axis=1)
flows_of_20100617['typeAttack'] = flows_of_20100617.Tag.replace(['Attack', 'Normal'], ['Brute Force SSH', 'N/A'])
flows_of_20100617['date'] = '20100617'

# all_flows = flows_of_20100612 \
#     .append(flows_of_20100613, sort=True) \
#     .append(flows_of_20100614, sort=True) \
#     .append(flows_of_20100615, sort=True) \
#     .append(flows_of_20100616, sort=True) \
#     .append(flows_of_20100617, sort=True)

all_flows = flows_of_20100613 \
    .append(flows_of_20100614, sort=True) \
    .append(flows_of_20100615, sort=True) \
    .append(flows_of_20100617, sort=True)

all_flows = all_flows.reset_index(drop=True)

# Guardamos los flujos en bruto en formato pickle
# flows_of_20100612.to_pickle('data/bulk_flows/flows_of_20100612.pickle')
# flows_of_20100613.to_pickle('data/bulk_flows/flows_of_20100613.pickle')
# flows_of_20100614.to_pickle('data/bulk_flows/flows_of_20100614.pickle')
# flows_of_20100615.to_pickle('data/bulk_flows/flows_of_20100615.pickle')
# flows_of_20100616.to_pickle('data/bulk_flows/flows_of_20100616.pickle')
# flows_of_20100617.to_pickle('data/bulk_flows/flows_of_20100617.pickle')
all_flows.to_pickle('data/bulk_flows/all_flows.pickle')

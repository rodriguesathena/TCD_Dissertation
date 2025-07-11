import numpy as np
import pandas as pd
import os

### STOP TEMPLATES ###
BRI_to_BRO = ['BRI', 'CHE', 'LAU', 'CCK', 'BAW', 'LEO', 'GAL', 'GLE', 'CPK', 'SAN', 'STI', 'KIL', 'BAL', 'DUN', 'WIN',
              'MIL', 'COW', 'BEE', 'RAN', 'CHA', 'HAR', 'STS', 'DAW', 'WES', 'OGP', 'OUP', 'DOM', 'BRD', 'GRA', 'PHI',
              'CAB', 'BRO']
SAN_to_BRO = ['SAN', 'STI', 'KIL', 'BAL', 'DUN', 'WIN', 'MIL', 'COW', 'BEE', 'RAN', 'CHA', 'HAR', 'STS', 'DAW', 'WES',
              'OGP', 'OUP', 'DOM', 'BRD', 'GRA', 'PHI', 'CAB', 'BRO']
BRI_to_PAR = ['BRI', 'CHE', 'LAU', 'CCK', 'BAW', 'LEO', 'GAL', 'GLE', 'CPK', 'SAN', 'STI', 'KIL', 'BAL', 'DUN', 'WIN',
              'MIL', 'COW', 'BEE', 'RAN', 'CHA', 'HAR', 'STS', 'DAW', 'WES', 'OGP', 'OUP', 'PAR']
SAN_to_PAR = ['SAN', 'STI', 'KIL', 'BAL', 'DUN', 'WIN', 'MIL', 'COW', 'BEE', 'RAN', 'CHA', 'HAR', 'STS', 'DAW', 'WES',
              'OGP', 'OUP', 'PAR']
BRO_to_BRI = ['BRO', 'CAB', 'PHI', 'GRA', 'BRD', 'DOM', 'PAR', 'MAR', 'TRY', 'DAW', 'STS', 'HAR', 'CHA', 'RAN', 'BEE',
              'COW', 'MIL', 'WIN', 'DUN', 'BAL', 'KIL', 'STI', 'SAN', 'CPK', 'GLE', 'GAL', 'LEO', 'BAW', 'CCK', 'LAU',
              'CHE', 'BRI']
PAR_to_BRI = ['PAR', 'MAR', 'TRY', 'DAW', 'STS', 'HAR', 'CHA', 'RAN', 'BEE', 'COW', 'MIL', 'WIN', 'DUN', 'BAL', 'KIL',
              'STI', 'SAN', 'CPK', 'GLE', 'GAL', 'LEO', 'BAW', 'CCK', 'LAU', 'CHE', 'BRI']
BRO_to_SAN = ['BRO', 'CAB', 'PHI', 'GRA', 'BRD', 'DOM', 'PAR', 'MAR', 'TRY', 'DAW', 'STS', 'HAR', 'CHA', 'RAN', 'BEE',
              'COW', 'MIL', 'WIN', 'DUN', 'BAL', 'KIL', 'STI', 'SAN']
PAR_to_SAN = ['PAR', 'MAR', 'TRY', 'DAW', 'STS', 'HAR', 'CHA', 'RAN', 'BEE', 'COW', 'MIL', 'WIN', 'DUN', 'BAL', 'KIL',
              'STI', 'SAN']
BRI_to_STS = ['BRI', 'CHE', 'LAU', 'CCK', 'BAW', 'LEO', 'GAL', 'GLE', 'CPK', 'SAN', 'STI', 'KIL', 'BAL', 'DUN', 'WIN',
              'MIL', 'COW', 'BEE', 'RAN', 'CHA', 'HAR', 'STS']
SAN_to_STS = ['SAN', 'STI', 'KIL', 'BAL', 'DUN', 'WIN','MIL', 'COW', 'BEE', 'RAN', 'CHA', 'HAR', 'STS']
BRO_to_DOM = ['BRO', 'CAB', 'PHI', 'GRA', 'BRD', 'DOM']
SAG_to_CON = ['SAG', 'FOR', 'CIT', 'CVN', 'FET', 'BEL', 'KIN', 'RED', 'KYL', 'BLU', 'BLA', 'DRI', 'GOL', 'SUI', 'RIA',
              'FAT', 'JAM', 'MUS', 'SMI', 'FOU', 'ABB', 'BUS', 'CON']
TAL_to_CON = ['TAL', 'HOS', 'COO', 'BEL', 'KIN', 'RED', 'KYL', 'BLU', 'BLA', 'DRI', 'GOL', 'SUI', 'RIA', 'FAT', 'JAM',
              'MUS', 'SMI', 'FOU', 'ABB', 'BUS', 'CON']
SAG_to_TPT = ['SAG', 'FOR', 'CIT', 'CVN', 'FET', 'BEL', 'KIN', 'RED', 'KYL', 'BLU', 'BLA', 'DRI', 'GOL', 'SUI', 'RIA',
              'FAT', 'JAM', 'MUS', 'SMI', 'FOU', 'ABB', 'GDK', 'MYS', 'SDK', 'TPT']
TAL_to_TPT = ['TAL', 'HOS', 'COO', 'BEL', 'KIN', 'RED', 'KYL', 'BLU', 'BLA', 'DRI', 'GOL', 'SUI', 'RIA', 'FAT', 'JAM',
              'MUS', 'SMI', 'FOU', 'ABB', 'GDK', 'MYS', 'SDK', 'TPT']
CON_to_SAG = ['CON', 'BUS', 'ABB', 'JER', 'FOU', 'SMI', 'MUS', 'HEU', 'JAM', 'RIA', 'SUI', 'GOL', 'DRI', 'BLU', 'KYL',
              'RED', 'KIN', 'BEL', 'FET', 'CVN', 'CIT', 'FOR', 'SAG']
TPT_to_SAG = ['TPT', 'SDK', 'MYS', 'GDK', 'ABB', 'JER', 'FOU', 'SMI', 'MUS', 'HEU', 'JAM', 'RIA', 'SUI', 'GOL', 'DRI',
              'BLU', 'KYL', 'RED', 'KIN', 'BEL', 'FET', 'CVN', 'CIT', 'FOR', 'SAG']
CON_to_TAL = ['CON', 'BUS', 'ABB', 'JER', 'FOU', 'SMI', 'MUS', 'HEU', 'JAM', 'RIA', 'SUI', 'GOL', 'DRI', 'BLU', 'KYL',
              'RED', 'KIN', 'BEL', 'COO', 'HOS', 'TAL']
TPT_to_TAL = ['TPT', 'SDK', 'MYS', 'GDK', 'ABB', 'JER', 'FOU', 'SMI', 'MUS', 'HEU', 'JAM', 'RIA', 'SUI', 'GOL', 'DRI',
              'BLU', 'KYL', 'RED', 'KIN', 'BEL', 'COO', 'HOS', 'TAL']
CON_to_HEU = ['CON', 'BUS', 'ABB', 'JER', 'FOU', 'SMI', 'MUS', 'HEU']
CON_to_RED = ['CON', 'BUS', 'ABB', 'JER', 'FOU', 'SMI', 'MUS', 'HEU', 'JAM', 'RIA', 'SUI', 'GOL', 'DRI', 'BLU', 'KYL',
              'RED']
SAG_to_BEL = ['SAG', 'FOR', 'CIT', 'CVN', 'FET', 'BEL']
SAG_to_BLA = ['SAG', 'FOR', 'CIT', 'CVN', 'FET', 'BEL', 'KIN', 'RED', 'KYL', 'BLU', 'BLA']
TAL_to_BEL = ['TAL', 'HOS', 'COO', 'BEL']
TAL_to_BLA = ['TAL', 'HOS', 'COO', 'BEL', 'KIN', 'RED', 'KYL', 'BLU', 'BLA']
TPT_to_HEU = ['TPT', 'SDK', 'MYS', 'GDK', 'ABB', 'JER', 'FOU', 'SMI', 'MUS', 'HEU']
TPT_to_RED = ['TPT', 'SDK', 'MYS', 'GDK', 'ABB', 'JER', 'FOU', 'SMI', 'MUS', 'HEU', 'JAM', 'RIA', 'SUI', 'GOL', 'DRI',
              'BLU', 'KYL', 'RED']
stop_templates = {
    ('BRI', 'BRO'): BRI_to_BRO, ('SAN', 'BRO'): SAN_to_BRO, ('BRI', 'PAR'): BRI_to_PAR, ('SAN', 'PAR'): SAN_to_PAR,
    ('BRO', 'BRI'): BRO_to_BRI, ('BRO', 'SAN'): BRO_to_SAN, ('PAR', 'BRI'): PAR_to_BRI, ('PAR', 'SAN'): PAR_to_SAN,
    ('BRI', 'STS'): BRI_to_STS,  ('SAN', 'STS'): SAN_to_STS,('BRO', 'DOM'): BRO_to_DOM,
    ('SAG', 'CON'): SAG_to_CON, ('TAL', 'CON'): TAL_to_CON, ('SAG', 'TPT'): SAG_to_TPT, ('TAL', 'TPT'): TAL_to_TPT,
    ('CON', 'SAG'): CON_to_SAG, ('TPT', 'SAG'): TPT_to_SAG, ('CON', 'TAL'): CON_to_TAL, ('TPT', 'TAL'): TPT_to_TAL,
    ('CON','HEU'):  CON_to_HEU, ('CON', 'RED'): CON_to_RED, ('SAG', 'BEL'): SAG_to_BEL, ('SAG','BLA'):  SAG_to_BLA, 
    ('TAL','BEL'):  TAL_to_BEL, ('TAL', 'BLA'): TAL_to_BLA, ('TPT', 'HEU'): TPT_to_HEU, ('TPT','RED'):  TPT_to_RED}

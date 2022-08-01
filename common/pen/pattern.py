import regex as re

NUMBER_OR_FRACTION_PATTERN = re.compile('^([+\\-]?(?:\\d+/\\d+|(?:\\d{1,3}(?:,\\d{3})+|\\d+)(?:\\.\\d+)?))')
ORDINAL_PATTERN = re.compile('^(\\d+)(st|nd|rd|th)$')
PUNCTUATION_END_PATTERN = re.compile('\\p{Punct}+$')

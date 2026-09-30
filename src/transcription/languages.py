"""Language metadata without importing model packages. Recognition is not translation."""

LANGUAGES = dict(item.split(':', 1) for item in (
    'en:English fa:Persian/Farsi ar:Arabic tr:Turkish de:German fr:French '
    'es:Spanish ru:Russian zh:Chinese ko:Korean ja:Japanese pt:Portuguese '
    'pl:Polish ca:Catalan nl:Dutch sv:Swedish it:Italian id:Indonesian hi:Hindi '
    'fi:Finnish vi:Vietnamese he:Hebrew uk:Ukrainian el:Greek ms:Malay cs:Czech '
    'ro:Romanian da:Danish hu:Hungarian ta:Tamil no:Norwegian th:Thai ur:Urdu '
    'hr:Croatian bg:Bulgarian lt:Lithuanian la:Latin mi:Maori ml:Malayalam cy:Welsh '
    'sk:Slovak te:Telugu lv:Latvian bn:Bengali sr:Serbian az:Azerbaijani sl:Slovenian '
    'kn:Kannada et:Estonian mk:Macedonian br:Breton eu:Basque is:Icelandic hy:Armenian '
    'ne:Nepali mn:Mongolian bs:Bosnian kk:Kazakh sq:Albanian sw:Swahili gl:Galician '
    'mr:Marathi pa:Punjabi si:Sinhala km:Khmer sn:Shona yo:Yoruba so:Somali af:Afrikaans '
    'oc:Occitan ka:Georgian be:Belarusian tg:Tajik sd:Sindhi gu:Gujarati am:Amharic '
    'yi:Yiddish lo:Lao uz:Uzbek fo:Faroese ht:Haitian-Creole ps:Pashto tk:Turkmen '
    'nn:Nynorsk mt:Maltese sa:Sanskrit lb:Luxembourgish my:Myanmar bo:Tibetan '
    'tl:Tagalog mg:Malagasy as:Assamese tt:Tatar haw:Hawaiian ln:Lingala ha:Hausa '
    'ba:Bashkir jw:Javanese su:Sundanese'
).split())
WHISPER_LANGUAGES = list(LANGUAGES)
ALIASES = {**{name.lower(): code for code, name in LANGUAGES.items()},
           'persian': 'fa', 'farsi': 'fa', 'فارسی': 'fa', 'fas': 'fa', 'per': 'fa',
           'eng': 'en', 'jv': 'jw', 'burmese': 'my', 'mandarin': 'zh'}


def normalize_language(value=None):
    value = str(value or 'auto').strip().lower().replace('_', '-')
    if value in ('auto', 'detect', ''):
        return None
    value = ALIASES.get(value, value.split('-')[0])
    if value not in LANGUAGES:
        raise ValueError('Unknown speech language: ' + value)
    return value


def language_options():
    return [{'code': 'auto', 'name': 'Detect language automatically'}] + [
        {'code': code, 'name': name, 'rtl': code in ('fa', 'ar', 'he', 'ur', 'ps', 'sd', 'yi')}
        for code, name in sorted(LANGUAGES.items(), key=lambda pair: (pair[0] not in ('fa', 'en'), pair[1]))]

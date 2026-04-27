# Copyright (C) 2019 FireEye, Inc. All Rights Reserved.

import re
import os
import json
import math
import numpy
import base64
import joblib
import string
import binascii
import fasttext
import unicodedata
import collections
import sklearn.pipeline
import sklearn.feature_extraction.text
from sklearn.base import BaseEstimator, TransformerMixin


if __package__ is None or __package__ == "":
    from lib import util
    from lib import stats
else:
    from .lib import util
    from .lib import stats

# preload from lib
dirname = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(dirname, 'lib/constants.json'), 'rb') as fid:
    constants = {k: set(v) for k, v in json.load(fid).items()}

with util.redirect_stderr():
    lid_model = fasttext.load_model(os.path.join(dirname, 'lib/lid.176.ftz'))
markov_model = joblib.load(os.path.join(dirname, 'lib/markov.pkl'))
log_transition_probas = markov_model['transition_matrix']
char_idx_mapper = markov_model['key_to_idx_map']


class Mapper(BaseEstimator, TransformerMixin):
    def __init__(self, func):
        self.func = func

    def fit(self, X, y=None):
        pass

    def transform(self, X):
        return numpy.array(list(map(self.func, X))).reshape(-1, 1)

    def get_feature_names(self):
        pass


class Featurizer():
    def __init__(self):
        self.b64chars = set(string.ascii_letters + string.digits + '+/_-=')
        dnsroot_cache = list(constants['dnsroot tlds']) + \
                        ['bit', 'dev', 'onion']
        self.tldstr = '|'.join(dnsroot_cache)

        self.mac_only_regex = \
            re.compile(r"""
                ^
                (?:[A-Fa-f0-9]{2}:){5}
                [A-Fa-f0-9]{2}
                $
                """, re.VERBOSE)

        fqdn_base = r'(([a-z0-9_-]{1,63}\.){1,10}(%s))' % self.tldstr
        fqdn_str = fqdn_base + r'(?:\W|$)'
        self.fqdn_strict_only_regex = re.compile(r'^' + fqdn_base + r'$', re.I)
        self.fqdn_regex = re.compile(fqdn_str, re.I)
        self.email_valid = re.compile(r'([a-z0-9_\.\-+]{1,256}@%s)' % fqdn_base, re.I)

        _u8 = r'(?:[1-9]?\d|1\d\d|2[0-4]\d|25[0-5])'
        _ipv4pre = r'(?:[^\w.]|^)'
        _ipv4suf = r'(?=(?:[^\w.]|\.(?:\W|$)|$))'
        ip_base = r'((?:%s\.){3}%s)' % (_u8, _u8)
        self.ip_regex = re.compile(_ipv4pre + ip_base + _ipv4suf)

        svc_base = r'(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?):[0-9]{1,5}'
        self.svc_regex = re.compile(svc_base)
        self.md5_only_regex = re.compile(r'^[A-Fa-f0-9]{32}$')
        self.sha1_only_regex = re.compile(r'^[A-Fa-f0-9]{40}$')
        self.sha256_only_regex = re.compile(r'^[A-Fa-f0-9]{64}$')
        self.url_regex = re.compile(r'\w+://[^ \'"\t\n\r\f\v]+')
        self.pkcs_regex = re.compile(r'-----BEGIN ([a-zA-Z0-9 ]+)-----')
        self.format_regex = re.compile(r'%[-|\+|#|0]?([\*|0-9])?(\.[\*|0-9])?[h|l|j|z|t|L]?[diuoxXfFeEgGaAcspn%]')
        self.linefeed_regex = re.compile(r'\\\\n$')
        self.path_regex = re.compile(r'[A-Z|a-z]\:\\\\[A-Za-z0-9]')
        self.pdb_regex = re.compile(r'\w+\.pdb\b')
        self.guid_regex = re.compile(r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89ab][0-9a-fA-F]{3}-[0-9a-fA-F]{12}')
        self.event_regex = re.compile(r'On(?!User|Board|Media|Global)(?:[A-Z][a-z]+)+')
        self.keylogger_regex = re.compile(r'\[[A-Za-z0-9\_\-\+ ]{2,13}\]')
        self.oid_regex = re.compile(r'((0\.([0-5]|9))|(1\.[0-3])|(2\.(([0-2][0-8])|(4[0-2])|(4[8-9])|(5[0-2])|(999))))(\.[0-9])+')
        self.ext_regex = re.compile(r'\w+\.[a-z]{3,4}\b')
        self.prod_id_regex = re.compile(r'[0-9]{5}-[0-9A-Z]{3}-[0-9]{7}-[0-9]{5}')
        self.priv_regex = re.compile(r'Se[A-Z][A-z]+Privilege')
        self.sddl_regex = re.compile(r'[DSO]:.+;;;.+$')
        self.sid_regex = re.compile(r'S-(?:[0-5]|9|(11)|(12)|(16))-')
        self.whitespace_regex = re.compile(r'\s+')
        self.letters_regex = re.compile(r'[^A-Za-z]')
        self.english_ignores = constants['windows api'].union(constants['pma important functions']).union\
                               (constants['dates']).union(constants['languages'])
        self.not_latin_unicode_names = ['ARABIC', 'SYRIAC', 'CYRILLIC', 'CJK', 'GEORGIAN']
        self.uppercase_var_name = re.compile(r'(?:\(| |^)[A-Z]+(?:\_[A-Z]+)+(?:\)| |$)')
        self.period_delimited_var_name = re.compile(r'(?:\(| |^)[a-z]{2,}(?:\.[a-z]{2,})+(?:\)| |$)')
        self.oss_substr_regex = re.compile(
            r'^(?:NT(?:3\.1|3\.5|3\.51))|' +
            r'(?:Ultimate(?: N| Edition|\_Edition))|' +
            r'(?:Business(?: N| Edition|\_Edition))|' +
            r'(?:Professional(?: Edition| x64 Edition))|' +
            r'(?:Microsoft Windows (?:ME|95|98|2000|XP))|' +
            r'(?:Storage(?: Server 2003 R2| Server 2003))|' +
            r'(?:Server(?: 2008| 2003| 2003 R2|2008|2008R2))|' +
            r'(?:Windows\+(?:2000|Home Server|Vista|Server\+2003|7|8|XP|8\.1))|' +
            r'(?:WIN(?:32\_NT|\_2008R2|\_7|\_2008|\_VISTA|\_2003|\_XPe|\_XP|\_2000))|' +
            r'(?:(?:Small\_Business\_|Small Business |Advanced )Server)|(?:Windows Storage Server 2003)|' +
            r'(?:Windows (?:7 \(6\.1\)|2000|Me|98|95|NT|Vista|7|8|10|XP|8\.1|Server|Home Server))|' +
            r'(?:Windows Server (?:2012 R2|2003|2003 R2|2008|2008 R2|R2|2000|2012|2003 R2 \(5\.2\)|2008 \(6\.0\)|2008 R2 \(6\.1\)))|' +
            r'(?:Standard(?:\_Edition|\_Edition\_core\_installation| Edition| Edition\(Core\)| x64 Edition| Edition \(core installation\)))|' +
            r'(?:Win(?:8|7|Server2003R2|Server2003|2K|XP64|XP| XP| 2000|HomeServer|NT|Server2012|Server2008R2|Server2008| Vista| Srv 2008| 7| 8| Srv 2003| Srv| ))|' +
            r'(?:Datacenter(?: Edition\(Core\)| Server| Edition for Itanium\-based Systems| x64 Edition| Edition| Edition \(core installation\)|\_Edition\_core\_installation|\_Edition))|' +
            r'(?:Home(?: Premium N| Premium| Basic N| Basic| Edition| Basic Edition| Premium Edition|\_Premium\_Edition|\_Basic\_Edition|\_Server|\-Premium\-Edition|\-Basic\-Edition|\_Edition))|' +
            r'(?:Enterprise(?:\_Edition|\_Edition\_for\_ItaniumBased\_System|\_Edition\_core\_installation| N| Edition| Edition\(Core\)| x64 Edition| Edition \(core installation\)| Edition for Itanium\-based Systems))|' +
            r'(?:(?:Small\_Business\_Server\_Premium\_|Small Business Server Premium |Small\_Business\_Server\_Premium\_Edition|Web\_Server\_|Cluster Server |Starter |Starter\_|Cluster\_Server\_|32\-bit |64\-bit |Embedded |Tablet PC |Media Center |Web |Compute Cluster |Web Server )Edition)$')
        self.oss_exact_regex = re.compile(r'^(?:2008|2003|2000|Business|Ultimate|Vista|Seven|Professional)$')
        self.user_agents_regex = re.compile(r'[\w\-]+\/[\w\-]+\.[\w\-]+(?:\.[\w\-])* ?(?:\[[a-z]{2}\] )?\((?:.+[:;\-].+|[+ ]?http://.+)\)')
        self.hive_regex = re.compile(r'[^0-9a-zA-Z](?:hkcu|hklm|hkey\_current\_user|hkey\_local\_machine)[^0-9a-zA-Z]')
        self.namespace_regex = re.compile(r'\\\\\.\\.*')
        self.msword_regex = re.compile(r'Word\.Document')
        self.mozilla_api_regex = re.compile(r'PR\_(?:[A-Z][a-z]{2,})+')
        self.privilege_constant_regex = re.compile(r'SE\_(?:[A-Z]+\_)+NAME')
        self.upx_regex = re.compile(r'\b(?:[a-z]?upx|[A-Z]?UPX)(?:\d|\b)')
        self.crypto_common_regex = re.compile(r'\b(?:rsa|aes|rc4|salt|md5)\b')
        self.features = [
            'string_length',
            'has_english_text',
            'entropy_rate',
            'english_letter_freq_div',
            'average_scrabble_score',
            'whitespace_percentage',
            'alpha_percentage',
            'digit_percentage',
            'punctuation_percentage',
            'vowel_consenant_ratio',
            'capital_letter_ratio',
            'title_words_ratio',
            'average_word_length',
            'has_ip',
            'has_ip_srv',
            'has_url',
            'has_email',
            'has_fqdn',
            'has_namespace',
            'has_msword_version',
            'has_packer',
            'has_crypto_related',
            'is_blacklisted',
            'has_privilege_constant',
            'has_mozilla_api',
            'is_strict_fqdn',
            'has_hive_name',
            'is_mac',
            'has_extension',
            'is_md5',
            'is_sha1',
            'is_sha256',
            'is_irrelevant_windows_api',
            'has_guid',
            'is_antivirus',
            'is_whitelisted',
            'is_common_dll',
            'is_boost_lib',
            'is_delphi_lib',
            'has_event',
            'is_registry',
            'has_malware_identifier',
            'has_sid',
            'has_keylogger',
            'has_oid',
            'has_product_id',
            'is_oss',
            'is_user_agent',
            'has_sddl',
            'has_protocol',
            'is_protocol_method',
            'is_base64',
            'is_hex_not_numeric_not_alpha',
            'has_format_specifier',
            'ends_with_line_feed',
            'has_path',
            'has_pdb',
            'has_privilege',
            'is_known_xml',
            'is_cpp_runtime',
            'is_library',
            'is_date',
            'is_pe_artifact',
            'has_public_key',
            'markov_junk',
            'is_x86',
            'is_common_path',
            'is_code_page',
            'is_language',
            'is_region_tag',
            'has_not_latin',
            'is_known_folder',
            'is_malware_api',
            'is_environment_variable',
            'has_variable_name',
            'has_padding_string'
        ]

    def _substring_match_bool(self, string_i, corpus):
        pass

    def _exact_match_bool(self, string_i, corpus):
        pass

    def string_length(self, string_i):
        pass

    def has_english_text(self, string_i, thresh_upper=0.9):
        pass

    def entropy_rate(self, string_i, base=2,
                     thresh_upper=3.65, thresh_lower=1.45):
        pass

    def english_letter_freq_div(self, string_i, thresh_upper=3.0):
        """
        estimated KL divergence from english letter distribution
        (case insensitive). Non-alpha bytes are ignored
         low KL divergence <=> letter freqs similar to English
        """
        pass

    def average_scrabble_score(self, string_i, thresh_lower=1.,
                               thresh_upper=3.51):
        pass

    def whitespace_percentage(self, string_i):
        pass

    def alpha_percentage(self, string_i):
        pass

    def digit_percentage(self, string_i):
        pass

    def punctuation_percentage(self, string_i):
        pass

    def vowel_consenant_ratio(self, string_i):
        pass

    def capital_letter_ratio(self, string_i):
        pass

    def title_words_ratio(self, string_i):
        pass

    def average_word_length(self, string_i):
        pass

    def has_ip_srv(self, string_i):
        pass

    def is_base64(self, string_i):
        # known FPs
        pass

    def is_hex_not_numeric_not_alpha(self, string_i):
        pass

    def is_strict_fqdn(self, string_i):
        pass

    def has_email(self, string_i):
        pass

    def is_md5(self, string_i):
        pass

    def is_sha1(self, string_i):
        pass

    def is_sha256(self, string_i):
        pass

    def is_mac(self, string_i):
        pass

    def has_keylogger(self, string_i):
        pass

    def has_oid(self, string_i):
        pass

    def has_privilege(self, string_i):
        pass

    def has_sddl(self, string_i):
        pass

    def has_mozilla_api(self, string_i):
        pass

    def is_oss(self, string_i):
        pass

    def has_packer(self, string_i):
        pass

    def has_crypto_related(self, string_i):
        pass

    def is_blacklisted(self, string_i):
        pass

    def has_namespace(self, string_i):
        pass

    def has_msword_version(self, string_i):
        pass

    def has_privilege_constant(self, string_i):
        pass

    def has_fqdn(self, string_i):
        pass

    def has_product_id(self, string_i):
        pass

    def has_ip(self, string_i):
        pass

    def has_sid(self, string_i):
        pass

    def has_url(self, string_i):
        pass

    def ends_with_line_feed(self, string_i):
        pass

    def has_path(self, string_i):
        pass

    def has_event(self, string_i):
        pass

    def has_guid(self, string_i):
        pass

    def has_public_key(self, string_i):
        pass

    def has_pdb(self, string_i):
        pass

    def is_user_agent(self, string_i):
        pass

    def has_hive_name(self, string_i):
        pass

    def has_variable_name(self, string_i):
        pass

    def has_format_specifier(self, string_i):
        pass

    def has_extension(self, string_i):
        pass

    def has_padding_string(self, string_i):
        pass

    def has_malware_identifier(self, string_i):
        pass

    def is_registry(self, string_i):
        pass

    def is_antivirus(self, string_i):
        pass

    def is_whitelisted(self, string_i):
        pass

    def has_protocol(self, string_i):
        pass

    def is_protocol_method(self, string_i):
        pass

    def is_common_path(self, string_i):
        pass

    def is_common_dll(self, string_i):
        pass

    def is_boost_lib(self, string_i):
        pass

    def is_delphi_lib(self, string_i):
        pass

    def is_irrelevant_windows_api(self, string_i):
        pass

    def is_cpp_runtime(self, string_i):
        pass

    def is_library(self, string_i):
        pass

    def is_date(self, string_i):
        pass

    def is_known_xml(self, string_i):
        pass

    def is_pe_artifact(self, string_i):
        pass

    def is_language(self, string_i):
        pass

    def is_code_page(self, string_i):
        pass

    def is_region_tag(self, string_i):
        pass

    def is_known_folder(self, string_i):
        pass

    def is_malware_api(self, string_i):
        pass

    def is_environment_variable(self, string_i):
        pass

    def is_x86(self, string_i):
        pass

    def has_not_latin(self, string_i):
        pass

    def markov_junk(self, string_i, thresh_lower=0.004):
        pass

    def _two_gram(self, string_i):
        pass

import re, os, csv

#from flask import request, render_template
#from markupsafe import Markup
#from bs4 import BeautifulSoup

#try:
#    from karamshoev import app
#except ModuleNotFoundError:
#    pass
#try:
#    import BeautifulSoup
#except ModuleNotFoundError:
#    pass

current_folder = os.path.dirname(os.path.abspath(__file__))
ortho_file_path = os.path.join(current_folder, os.path.join("static", "ortho.txt"))
ortho_table_path = os.path.join(current_folder, os.path.join("static", "ortho_table.csv"))

problem_symbols = ("gamma", "gh", "j", "sh")
cyr_check = "[ПпБДдЛлЖжШшЩщФфЦцЧчИиЙйЬьЪъЫыЭэЮюЯя]"
cyr_precise_check = "[аāбвгдежзиӣйклмнопрстуӯфхцчшщъьыэюяқғҳҷ]"
lat_check = "[SsVvFfIiGgZzQqNR]"
lat_precise_check = "[aābcdefghiījklmnopqrstuūůvxyzčžš]"

_cyr = ("cyr", "cyrillic", "кир", "кириллица")
_lat = ("lat", "latin", "лат", "латиница")


def detect_orthography(text, lang="sgh"):
    return Converter(target="orig", settings="auto", lang=lang).convert(text).base


def normalize(text, lang="sgh"):
    return Converter(target="orig", settings="auto", lang=lang).convert(text).text


def isnormal(text, lang="sgh"):
    normalized = normalize(text, lang=lang)
    return normalized == text


class ConverterOutput:
    """The output of the Converter.convert() function.
    """
    
    def __init__(self, text, base, code, settings, base_fixed=False):
        """
        Parameters
        ----------
        text : str
            The converted text
        base : str
            The base of the orthography: "lat" or "cyr"
        code : dict
            The dictionary with the correspondences of tricky symbols that were used in the convertion
        settings : dict
            The dictionary with the settings of the convertion
        base_fixed : bool
            Whether problems were detected and automatically fixed in the base orthography
            (Cyrillic letters in a predominantly Latin text or vice versa)
        """
        
        self.text = text
        self.base = base
        self.code = code
        self.settings = settings
        self.base_fixed = base_fixed

    def __repr__(self):
        return self.text

    def full(self):
        """Prints all the settings of the converted text.
        """
        return f"ConverterOutput(\n\ttext='{self.text}',\n\tbase='{self.base}',\n\tcode={self.code},\n\tsettings={self.settings},\n\tbase_fixed={self.base_fixed}\n)"


class Converter:
    """The main class to convert from alphabet X to alphabet Y.

    Make a `Converter` object. Optionally, set the default target alphabet as an argument:
    > c = Converter(target="cyr")
    
    Then use the method `convert` to change the alphabet of a text:
    > converted_text = c.convert(original_text)
    """
    
    def __init__(self, target="orig", settings="auto", lang="sgh", base=None,
                 gamma=None, gh=None, j=None, sh=None, accent=None, eqtohyphen=None):
        """
        Parameters
        ----------
        target : str, optional
            Default: "orig". The target orthography, one of the four options:
            "lat" (Latin), "cyr" (Cyrillic), "ipa" (IPA)
            or "orig" (leave the original orthography, whichever it is)
        settings : dict, optional
            Default: "auto". The dictionary with the settings or the string "auto"
            to indicate that the settings must be decided automatically (preferred)
            Example of settings:
            settings = {
                "gamma": "auto",
                "gh": "auto",
                "j": "auto",
                "sh": "auto",
                "accent": True,
                "eqtohyphen": False
            }
        lang : int, optional
            Default: "sgh". The language of the text
            Options: "sgh" (Shughni), "rush" (Rushani & Khufi),
            "bart" (Bartangi & Roshorvi)
        base: str, optional
            Default: None. The base orthography, one of the two options:
            "lat" (Latin), "cyr" (Cyrillic). If None, no base orthography is chosen
        gamma : str, optional
            Default: None. The phonetic meaning of the Greek gamma (Γγ).
            One of the two options: "velar" (IPA [ɣ]) or "uvular" (IPA [ʁ])
        gh : str, optional
            Default: None. The phonetic meaning of the Latin gamma (Ɣɣ).
            One of the two options: "velar" (IPA [ɣ]) or "uvular" (IPA [ʁ])
        j : str, optional
            Default: None. The phonetic meaning of the Latin letter Jj.
            One of the three options: "y" (IPA [j]), "dz" (IPA [dz]) or "dzh" (IPA [dʒ])
        sh : str, optional
            Default: None. The phonetic meaning of the "sh" combination.
            One of the two options: "sch" (IPA [ʃ]) or "skh" (IPA [sh])
        accent: bool, optional
            Default: True. Whether the accent symbol should be kept (True) or removed (False)
        eqtohyphen: bool, optional
            Default: False. Whether all equal sign ‘=’ should be changed to the hyphen sign ‘-’ (True) or not (False)
            One of the two options: True or False
        """
        
        with open(ortho_table_path, 'r', encoding='utf-8-sig') as f:  # encoding 'utf-8-sig' helps to get rid
            reader = csv.reader(f, delimiter=",")  # of some technical .csv symbols
            self._ortho = {}
            headers = []
            for row in reader:
                if row[0] == "src":
                    headers = row
                else:
                    self._ortho[row[0]] = {
                        headers[i]: row[i] for i in range(1, len(row))}
        
        with open(ortho_file_path, 'r', encoding='utf-8-sig') as f:  # encoding 'utf-8-sig' helps to get rid
            self._final_ortho = [x.strip("\n") for x in f.readlines()]  # of some technical .csv symbols
        
        self.lang = lang
        self.settings = {
            "gamma": "auto",
            "gh": "auto",
            "j": "auto",
            "sh": "auto",
            "accent": accent,
            "eqtohyphen": eqtohyphen
        } if settings == "auto" else settings
        if gamma:
            self.settings["gamma"] = gamma
        if gh:
            self.settings["gh"] = gh
        if j:
            self.settings["j"] = j
        if sh:
            self.settings["sh"] = sh
        if accent is not None:
            self.settings["accent"] = accent
        if eqtohyphen is not None:
            self.settings["eqtohyphen"] = eqtohyphen
        
        self.base = base
        self.target = target
        if target in _lat:
            self.lang_target = f"{self.lang}_{_lat[0]}"
        elif target in _cyr:
            self.lang_target = f"{self.lang}_{_cyr[0]}"
        elif target == "ipa":
            self.lang_target = f"{self.lang}_{target}"
        else:
            self.lang_target = self.target
        
        if self.target == "ipa":
            self.settings["accent"] = False

    def __repr__(self):
        return f"Converter(lang='{self.lang}', target='{self.target}', settings={self.settings})"
    
    def convert(self, text, target=None, base=None):
        """
        Converts a text.
        
        Parameters
        ----------
        text: str
            The text that will be converted
        target : str, optional
            Default: "lat". The target orthography, one of the three:
            "lat" (Latin), "cyr" (Cyrillic), "ipa" (IPA)
        """
        if text is None:
            return None
        if target is None:
            target = self.lang_target
        else:
            target = f"{self.lang}_{target}"
        
        if "accent" in self.settings:
            if self.settings["accent"] is False:
                text = re.sub("́", "", text)  # removing the accent
        if "eqtohyphen" in self.settings:
            if self.settings["eqtohyphen"] is True:
                text = re.sub("=", "-", text)  # changing the equal sign to hyphen
        text = re.sub("ǰ", "ǰ", text)    # replacing an especially tricky symbol
        
        auto_defined_base, base_fixed, code = None, False, {}
        if self.base and base is None:
            base = self.base
        if base:
            text_tokens = re.findall("\S+|\s+", text)
            base_tokens = []
            auto_defined_bases = []
            for i, token in enumerate(text_tokens):
                auto_defined_base, _ = definecode(token, self.settings)  # defining base and code of the script
                auto_defined_bases.append(auto_defined_base)
                if auto_defined_base == base:
                    base_tokens.append(token)
            _, code = definecode(" ".join(base_tokens), self.settings)

            for i, token in enumerate(text_tokens):
                auto_defined_base = auto_defined_bases[i]
                if auto_defined_base == base:
                    fixed_token = fixbase(auto_defined_base, token)  # fixing some problems (if necessary)
                    if fixed_token != token:
                        base_fixed = True
                    
                    text_tokens[i] = changecode(code, fixed_token, target, self._ortho, self._final_ortho,
                                                auto_defined_base, self.lang)  # converting symbols
            text = "".join(text_tokens)

        else:
            auto_defined_base, code = definecode(text, self.settings)  # defining base and code of the script
            
            fixed_text = fixbase(auto_defined_base, text)  # fixing some problems (if necessary)
            if fixed_text != text:
                base_fixed = True
            
            text = changecode(code, fixed_text, target, self._ortho, self._final_ortho, auto_defined_base, self.lang)  # converting symbols
        
        return ConverterOutput(text, auto_defined_base, code, self.settings, base_fixed)


def definecode(text, settings):
    """Defines base, aka cyrillic or latin script is used in the text, and what contradictory symbols mean"""

    # trying to define whether the text is written in cyrllic or latin script
    if len(text) > 20 or len(re.findall(" ", text)) > 3:
        cyr_sum = len(re.findall(cyr_check, text))
        lat_sum = len(re.findall(lat_check, text))
        coeff = 6

    else:
        cyr_sum = len(re.findall(cyr_precise_check, text.lower()))
        lat_sum = len(re.findall(lat_precise_check, text.lower()))
        coeff = 1

    if cyr_sum > lat_sum * coeff:
        base = "cyr"
    elif lat_sum > cyr_sum * coeff:
        base = "lat"
    else:
        base = "unknown"

    code = {}
    for symbol in problem_symbols:
        code[symbol] = "unknown"

    if len(re.findall("[Ss]h", text)) == 0:
        code["sh"] = "symbol not found"

    elif settings["sh"] == "auto":  # if the setting is to determine phonematic meaning of 'sh' automatically

        if len(re.findall("[ШшŠšɕ]", text)) == 0:  # if there is no different symbol which means 'sch' in the text
            code["sh"] = "sch"                     # sh must mean 'sch'

        else:                                      # if there is a different symbol which means 'sch' in the text
            code["sh"] = "skh"                     # sh must mean 'skh'

    else:
        code["sh"] = settings["sh"]  # if user has indicated how sh should be interpreted, this indication is used

    if len(re.findall("[Γγ][^̌]|[Γγ]$", text)) == 0:  # if symbol γ is not found in the text
        code["gamma"] = "symbol not found"

    elif settings["gamma"] == "auto":

        if len(re.findall("[Γγ]̌|[Гг]̌|[Ѓѓ][ЪъЬь]|[Ұұ]", text)) > 0:  # if there is a different symbol
            code["gamma"] = "uvular"                                # which means 'velar' in the text
                                                                    # γ must mean 'uvular'

        elif len(re.findall("[ЃѓʁҒғ]|[Ɣɣ][^̌]|[Ɣɣ]$", text)) > 0:  # if there is a different symbol
            code["gamma"] = "velar"                               # which means 'uvular' in the text
                                                                  # γ must mean 'velar'

        else:  # if there are no symbols for determination in the text
            code["gamma"] = "uvular"
    else:
        code["gamma"] = settings["gamma"]  # if user has indicated how γ should be interpreted, this indication is used

    if len(re.findall("[Ɣɣ][^̌]|[Ɣɣ]$", text)) == 0:  # if symbol ɣ is not found in the text
        code["gh"] = "symbol not found"

    elif settings["gh"] == "auto":
        # checking whether there are different symbols which mean 'gh' in the text
        isthere = {"other_velar": False, "gh_hachek": False, "other_uvular": False}

        # if there is a different symbol which means 'velar' in the text
        if len(re.findall("[Гг]̌|[Ѓѓ][ЪъЬь]|[Ұұ]", text)) > 0:
            isthere["other_velar"] = True

        # if there is symbol ɣ with hachek in the text
        if len(re.findall("[Ɣɣ]̌", text)) > 0:
            isthere["gh_hachek"] = True

        # if there is a different symbol which means 'uvular' in the text
        if len(re.findall("[ЃѓʁҒғ]", text)) > 0:
            isthere["other_uvular"] = True

        if isthere["other_uvular"] == False:
            code["gh"] = "uvular"

        elif isthere["gh_hachek"] == False and isthere["other_velar"] == False:
            code["gh"] = "velar"

    else:
        code["gh"] = settings["gh"]  # if user has indicated how ɣ should be interpreted,
                                             # this indication is used

    if len(re.findall("[Jj][^̌]|[Jj]$", text)) == 0:  # if symbol j is not found in the text
        code["j"] = "symbol not found"

    elif settings["j"] == "auto":

        if len(re.findall("[ЙйYy]", text)) == 0:  # if symbols й/y are not found, j must mean 'y'
            code["j"] = "y"

        elif len(re.findall("[Jj]̌|[ǰЉљҶҷ]", text)) == 0:  # if there is a different symbol which means 'dzh',
            code["j"] = "dzh"                             # j must mean 'dzh'

        elif len(re.findall("[ȤȥƷӡ]|[Зз][ъ̌]", text)) == 0:  # if there is a different symbol which means 'dz',
            code["j"] = "dz"                                # j must mean 'dz'
    else:
        code["j"] = settings["j"]  # if user has indicated how j should be interpreted,
                                   # this indication is used

    return base, code


def fixbase(base, text):
    """If the base is latin but there are random cyrillic letters in the text, they are fixed"""

    changer = [
        ["С", "C"],
        ["с", "c"],
        ["К", "K"],
        ["к", "k"],
        ["М", "M"],
        ["м", "m"],
        ["Н", "H"],
        ["Р", "P"],
        ["р", "p"],
        ["Т", "T"],
        ["т", "t"],
        ["У", "Y"],
        ["у", "y"],
        ["А", "A"],
        ["а", "a"],
        ["Е", "E"],
        ["е", "e"],
        ["В", "B"],
        ["О", "O"],
        ["о", "o"],
        ["Х", "X"],
        ["х", "x"]
    ]

    if base != "?":
        for symbol in changer:
            if base == "lat":
                text = re.sub(symbol[0], symbol[1], text)
            elif base == "cyr":
                text = re.sub(symbol[1], symbol[0], text)

    return text


def _bad_to_good(text, for_dict, lang, final_ortho):

    for line in final_ortho:

        if line.startswith("/"):
            if for_dict:
                continue
            else:
                line = line[1:]

        if line.startswith("@"):
            if lang != "sgh":
                continue
            else:
                line = line[1:]

        if not line.startswith('#'):  # lines with hash in ortho.txt are user notes
            bad, good = line.split(' ')
            text = re.sub(bad, good, text)

    return text


def changecode(code, text, lang_target, target_ortho, final_ortho, orig, lang, for_dict=False):
    """Converts the orthography"""

    if code["gamma"] == "velar":
        text = re.sub("Γ([^̌])", "Ɣ̌\1", text)  # replacing γ —> ɣ̌
        text = re.sub("γ([^̌])", "ɣ̌\1", text)

    if code["gh"] == "velar":
        text = re.sub("([Ɣɣ])([^̌])", r"\1̌\2", text)  # replacing ɣ —> ɣ̌

    if code["j"] == "y":
        text = re.sub("J([^̌])|J$", r"Y\1", text)  # replacing j —> y
        text = re.sub("j([^̌])|j$", r"y\1", text)

    if code["j"] == "dzh":
        text = re.sub("J([^̌])|J$", r"Ĵ\1", text)  # replacing j —> y
        text = re.sub("j([^̌])|j$", r"ĵ\1", text)

    elif code["j"] == "dz":
        text = re.sub("J([^̌])|J$", r"Ʒ\1", text)  # replacing j —> dz
        text = re.sub("j([^̌])|j$", r"ʒ\1", text)

    if code["sh"] == "sch":
        text = re.sub("S[Hh]", "Š", text)  # replacing sh —> š
        text = re.sub("sh", "š", text)
    
    if lang_target == "orig":
        if orig == "unknown":
            lang_target = f"{lang}_lat"
        else:
            lang_target = f"{lang}_{orig}"

    text = _bad_to_good(text=text, for_dict=for_dict, lang=lang, final_ortho=final_ortho)

    text = re.sub("̣", "", text)  #deleting Combining dot below

    for letter in target_ortho:
        text = re.sub(letter, target_ortho[letter][lang_target], text)

    return text


def fullconvert(text, target="lat", to_karamshoev=False, settings="auto", for_dict=False, lang="sgh"):
    """Manages all the functions"""

    if to_karamshoev:
        target="cyr"

    if settings == "auto":  # default settings
        settings = {
            "gamma": "auto",
            "gh": "auto",
            "j": "auto",
            "sh": "auto",
            "accent": True,
            "eqtohyphen": False
        }

    output = Converter(target=target, settings=settings, lang=lang).convert(text)

    return output.base, output.code, output.text


try:
    @app.route("/converter", methods=['GET', 'POST'])
    def converter():
        """Receives data from the html form, converts the text, and sends the result back to html interface"""

        upload_file = request.files["text_file"]  # trying to receive text from the possibly uploaded text file
        original_text = upload_file.read().decode("utf-8")  # decoding the text with Unicode

        if original_text == "":  # if file has not been uploaded
            original_text = request.values.get("text")  # trying to receive text from the text html field

        if original_text == "" or original_text is None:  # if the text is empty, nothing happens
            return render_template("ortho.html", convert_pack=False)

        ##########

        settings = {}
        for symbol in problem_symbols:
            settings[symbol] = request.values.get(symbol)  # receiving user settings
        target = request.values.get("dest")
        lang = request.values.get("lang")
        
        def booleize(stroka):
            return True if stroka in (True,"True") else False
        
        for setting in ("accent", "eqtohyphen"):
            settings[setting] = booleize(request.values.get(setting))
        
        base, code, converted_text = fullconvert(
            original_text, target=target, settings=settings, lang=lang)  # converting the text
        html = BeautifulSoup(converted_text, features='html.parser').get_text().replace('\n', '<br>')
        # list for more compact sending to the web
        convert_pack = {"base": base, "code": code, "original_text": original_text, "converted_text": converted_text, "converted_text_html": Markup(html)}

        return render_template("ortho.html", convert_pack=convert_pack)

except NameError:
    pass
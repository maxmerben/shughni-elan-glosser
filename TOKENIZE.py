import csv, os, pympi, re, datetime
import cortex as cx

morphemes_list_path = "morphemes_list.csv"

_gloss = "_morph"; _pos = "_pos"; _condition = "pos_condition"

#############################

shg_sentences_tiername = "sentence-txt-sgh"
shg_words_tiername     = "word-txt-sgh"
shg_morphs_tiername    = "morph-txt-sgh"
eng_glosses_tiername   = "morph-gls-en"
eng_pos_tiername       = "morph-pos-en"

#############################


def get_gloss_table():
    with (open(morphemes_list_path, "r", encoding="utf-8-sig") as f):
        reader = csv.reader(f, delimiter=",")

        _gloss_dict = {}
        prefixes = set()

        for row in reader:
            _gloss_dict[row[2]] = {
                _gloss:     row[3],
                _pos:       row[4],
                _condition: row[5]
            }
            if row[1] == "prefix":
                prefixes.add(cx.dehyphen(row[2]))
    return _gloss_dict, prefixes


def gloss_text(orig_filename: str):
    
    gloss_dict, prefixes = get_gloss_table()
    orig_filename = re.sub(".eaf$", "", orig_filename)
    g_counter, s_counter = 0, 0

    try:
        text = cx.Text(filename=orig_filename+".eaf")

        for sentence in text:

            sentence.tokens = [cx.Token(
                token=t, back_sent=sentence, back_text=text) for t in re.split("[ \t]+", sentence.text.strip())]
            for token in sentence.tokens:
                if re.fullmatch("[-=]+", token.token):
                    prefix, token_text = None, token.token
                else:
                    token_text = cx.depunct(token.token)
                    token.ana = [cx.Analysis(morphemes=[], back_token=token, back_sent=sentence, back_text=text)]

                    prefix = re.search("[^-=]+[-=]", token.token)

                    if prefix:
                        prefix = prefix[0]
                        if prefix[:-1].lower() in prefixes:
                            token_text = token_text[:len(prefix)]+"%" + re.sub(
                                "-", "%-", re.sub("=", "#=", token_text[len(prefix):])
                            )
                        else:
                            token_text = re.sub("-", "%-", re.sub("=", "#=", token_text))
                    else:
                        token_text = re.sub("-", "%-", re.sub("=", "#=", token_text))
                    
                    
                    morphs_split = re.split("[#%]", token_text)
                    for morph in morphs_split:
                        if len(morph) < 1:
                            continue
                        morph=cx.depunct(morph).lower()

                        try:
                            morph_for_search = re.sub("=", "-", morph)
                            gloss = gloss_dict[morph_for_search][_gloss]
                            if re.match(r"\?", gloss):
                                s_counter += 1
                            else:
                                g_counter += 1

                        except KeyError:
                            gloss = "?"
                            pass

                        token.ana[0].morphemes.append(cx.Morpheme(morph=morph, gloss=gloss,
                                            back_ana=token.ana[0], back_token=token,
                                            back_sent=sentence, back_text=text))

                    token.token = re.sub("-", "", token.token)
                    #token.token = re.sub("=", "-", token.token)
            
            sentence.text = re.sub("-", "", sentence.text)
            #sentence.text = re.sub("=", "-", sentence.text)

        text.numerate()

        print(f"> Текст токенизирован (количество токенов: {len(text.tokens)}).")
        print(f"> Токены разделены на морфемы (количество морфем: {len(text.morphemes)}).")
        print(f"> Добавлены англоязычные глоссы "
                f"(проставлены для {g_counter} морфем, "
                f"варианты предложены для {s_counter} морфем).")

        """
        # TOKENIZING INTO SHUGHNI MORPHS
        
        if len(eaf.tiers[shg_morphs_tiername_speaker][1]) == 0:

            counter = 0
        
            for w in eaf.tiers[shg_words_tiername_speaker][1].items():
                w_id = w[0]
                w_text = depunct(w[1][1])

                prefix = re.search("[^-=]+[-=]", w_text)

                if prefix:
                    prefix = prefix[0]
                    if prefix[:-1].lower() in prefixes:
                        w_text = w_text[:len(prefix)]+"%" + re.sub(
                            "-", "%-", re.sub("=", "#=", w_text[len(prefix):])
                        )
                    else:
                        w_text = re.sub("-", "%-", re.sub("=", "#=", w_text))
                else:
                    w_text = re.sub("-", "%-", re.sub("=", "#=", w_text))
                
                morphemization = re.split("[#%]", w_text)
                
                for i in range(len(morphemization)):
                    
                    if len(morphemization[i]) < 1:
                        continue
                    
                    morphemization[i] = depunct(morphemization[i]).lower()
                    
                    prev_id = None if i == 0 else list(eaf.tiers[shg_morphs_tiername_speaker][1].keys())[-1]
                    
                    aid = eaf.generate_annotation_id()
                    eaf.annotations[aid] = shg_morphs_tiername_speaker
                    eaf.tiers[shg_morphs_tiername_speaker][1][aid] = (w_id, morphemization[i], prev_id, None)

                    counter += 1

        # TOKENIZING INTO ENGLISH MORPH GLOSSES

        if len(eaf.tiers[eng_glosses_tiername_speaker][1]) == 0:
            s_counter = 0; g_counter = 0

            morphs = eaf.get_annotation_data_for_tier(shg_morphs_tiername_speaker)

            for m in eaf.tiers[shg_morphs_tiername_speaker][1].items():

                try:
                    morph = re.sub("=", "-", m[1][1])
                    gloss = gloss_dict[morph][_gloss]
                    if re.match(r"\?", gloss):
                        s_counter += 1
                    else:
                        g_counter += 1
                except KeyError:
                    gloss = None

                aid = eaf.generate_annotation_id()
                eaf.annotations[aid] = eng_glosses_tiername_speaker

                eaf.tiers[eng_glosses_tiername_speaker][1][aid] = (m[0], gloss, None, None)
    
        #################

        for id in eaf.tiers[shg_sentences_tiername_speaker][0].keys():
            
            eaf.tiers[shg_sentences_tiername_speaker][0][id] = (
                eaf.tiers[shg_sentences_tiername_speaker][0][id][0],
                eaf.tiers[shg_sentences_tiername_speaker][0][id][1],
                dehyphen(eaf.tiers[shg_sentences_tiername_speaker][0][id][2], equal=False),
                eaf.tiers[shg_sentences_tiername_speaker][0][id][3]
            )
        
        for id in eaf.tiers[shg_words_tiername_speaker][1].keys():
            
            eaf.tiers[shg_words_tiername_speaker][1][id] = (
                eaf.tiers[shg_words_tiername_speaker][1][id][0],
                dehyphen(eaf.tiers[shg_words_tiername_speaker][1][id][1], equal=False),
                eaf.tiers[shg_words_tiername_speaker][1][id][2],
                eaf.tiers[shg_words_tiername_speaker][1][id][3]
            )
        """

        t = re.sub(":", "", re.search("[^\.]+", datetime.datetime.now().isoformat())[0])
        new_filename = f"{orig_filename}_auto_{t}.eaf"
        text.to_eaf(filepath=new_filename, suffix="", update_eaf=True)

        print(f"\nГотово! Итоговый файл называется '{new_filename}'.\n")

        return new_filename
    
    except FileNotFoundError:
        print("\nОШИБКА! Такой eaf-файл не найден.\n")


def find_and_gloss_text(filename=None, interface=False):
    
    if interface:
        i = 0
        for f in os.listdir():
            if f.endswith(".eaf") and (re.search("auto", f) is None):
                i += 1
                x = f
        
        if i == 1:
            orig_filename = x
        
        else:
            orig_filename = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                input("Введите название файла .eaf: "))
    
        gloss_text(orig_filename)
    
    elif filename is not None and isinstance(filename, str):
        orig_filename = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), filename)
        
        gloss_text(orig_filename)


if __name__ == "__main__":
    
    while True:
        
        find_and_gloss_text(interface=True)
        
        goon = input("…Ещё один файл? Нажмите Enter.\nЕсли хотите выйти, напечатайте что угодно. ")
        if goon!="":
            break
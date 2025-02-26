import csv, os, pympi, re, datetime

_morph = "_morph"; _pos = "_pos"

#############################

shg_sentences_tiername = "phrase-txt-sgh"
shg_words_tiername     = "word-txt-sgh"
shg_morphs_tiername    = "morph-txt-sgh"
eng_glosses_tiername   = "morph-gls-en"
eng_pos_tiername       = "morph-pos-en"

punct = "[\.,\!\?…«»„“”\/\\:;'\"\[\]\(\)\{\}\<>@#\$\%\^\&\*\+]"

#############################


def gloss_text(orig_filename: str):
    
    if orig_filename.endswith(".eaf"):
        orig_filename = orig_filename[:-4]


    def dehyphen(text, equal=True):
        if equal:
            return re.sub("[\-−—=]+", "", text)
        else:
            return re.sub("[\-−—]+", "", text)


    def depunct(text):
        return re.sub(f"^{punct}+|{punct}+$", "", text)


    morphemes_list_path = "morphemes_list.csv"
    with (open(morphemes_list_path, "r", encoding="utf-8-sig") as f):
        reader = csv.reader(f, delimiter=",")

        morphs_list = {}
        prefixes = set()

        for row in reader:
            morphs_list[row[2]] = {
                _morph: row[3],
                _pos:   row[4]
            }
            if row[1] == "prefix":
                prefixes.add(dehyphen(row[2]))


    try:
        eaf = pympi.Elan.Eaf(file_path=f"{orig_filename}.eaf")
        
        
        for letter in ("A", "B", "C", "D"):
            shg_sentences_tiername_speaker = f"{letter}_{shg_sentences_tiername}"
            shg_words_tiername_speaker     = f"{letter}_{shg_words_tiername}"
            shg_morphs_tiername_speaker    = f"{letter}_{shg_morphs_tiername}"
            eng_glosses_tiername_speaker   = f"{letter}_{eng_glosses_tiername}"
            eng_pos_tiername_speaker       = f"{letter}_{eng_pos_tiername}"
        
        
            # SIGNALLING TIER MISTAKES

            try:
                sentences = eaf.get_annotation_data_for_tier(shg_sentences_tiername_speaker)
            except KeyError:
                if letter=="A":
                    print(f"\nОШИБКА! Слой '{shg_sentences_tiername_speaker}' не найден!")
                continue

            try:
                words = eaf.get_annotation_data_for_tier(shg_words_tiername_speaker)
            except KeyError:
                if letter=="A":
                    print(f"\nОШИБКА! Слой '{shg_words_tiername_speaker}' не найден!")
                continue
            
            print(f"\nГоворящ:ая {letter}:")
            
            # TOKENIZING INTO SHUGHNI TOKENS

            if len(eaf.tiers[shg_words_tiername_speaker][1]) == 0:
                counter = 0
            
                for s in eaf.tiers[shg_sentences_tiername_speaker][0].items():
                    
                    s_id = s[0]
                    s_text = s[1][2]
                    
                    tokenization = re.split("[ \t]+", s_text.strip())
                    
                    for i in range(len(tokenization)):
                        prev_id = None if i == 0 else list(eaf.tiers[shg_words_tiername_speaker][1].keys())[-1]
                        
                        aid = eaf.generate_annotation_id()
                        eaf.annotations[aid] = shg_words_tiername_speaker
                        eaf.tiers[shg_words_tiername_speaker][1][aid] = (s_id, tokenization[i], prev_id, None)
                        
                        counter += 1

                print(f"> Текст токенизирован (количество токенов: {counter}).")


            # TOKENIZING INTO SHUGHNI MORPHS
            
            if len(eaf.tiers[shg_morphs_tiername_speaker][1]) == 0:

                counter = 0
            
                for w in eaf.tiers[shg_words_tiername_speaker][1].items():
                    w_id = w[0]
                    w_text = depunct(w[1][1])

                    prefix = re.search("[^\-\=]+[\-\=]", w_text)

                    if prefix:
                        prefix = prefix[0]
                        if prefix[:-1].lower() in prefixes:
                            w_text = w_text[:len(prefix)]+"%" + re.sub(
                                "\-", "%-", re.sub("\=", "#=", w_text[len(prefix):])
                            )
                        else:
                            w_text = re.sub("\-", "%-", re.sub("\=", "#=", w_text))
                    else:
                        w_text = re.sub("\-", "%-", re.sub("\=", "#=", w_text))
                    
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

                print(f"> Токены разделены на морфемы (количество морфем: {counter}).")


            # TOKENIZING INTO ENGLISH MORPH GLOSSES

            if len(eaf.tiers[eng_glosses_tiername_speaker][1]) == 0:
                s_counter = 0; g_counter = 0

                morphs = eaf.get_annotation_data_for_tier(shg_morphs_tiername_speaker)

                for m in eaf.tiers[shg_morphs_tiername_speaker][1].items():

                    aid = eaf.generate_annotation_id()
                    eaf.annotations[aid] = eng_glosses_tiername_speaker

                    try:
                        gloss = morphs_list[re.sub("=", "-", m[1][1])][_morph]
                        if re.match("\?", gloss):
                            s_counter += 1
                        else:
                            g_counter += 1
                    except KeyError:
                        gloss = None

                    eaf.tiers[eng_glosses_tiername_speaker][1][aid] = (m[0], gloss, None, None)

                print(f"> Добавлены англоязычные глоссы "
                      f"(проставлены для {g_counter} морфем, "
                      f"варианты предложены для {s_counter} морфем).")
            

            # TOKENIZING INTO POS LABELS

            shg_morphs_items = list(eaf.tiers[shg_morphs_tiername_speaker][1].items())
            list_of_aids = []

            if len(shg_morphs_items) != 0:
                s_counter = 0; g_counter = 0

                if len(eaf.tiers[eng_pos_tiername_speaker][1].items()) > 0:
                    print(f"> Слой {eng_pos_tiername_speaker} очищен.")
                eaf.remove_all_annotations_from_tier(eng_pos_tiername_speaker)

                for i in range(len(shg_morphs_items)):

                    aid = eaf.generate_annotation_id()
                    eaf.annotations[aid] = eng_pos_tiername_speaker

                    try:
                        pos = morphs_list[re.sub("=", "-", shg_morphs_items[i][1][1])][_pos]
                        if (len(pos) == 0) or (re.match("< ", pos)):
                            i_pos = None
                        else:
                            i_pos = pos
                            if re.match("\?", i_pos):
                                s_counter += 1
                            else:
                                g_counter += 1

                    except KeyError:
                        pos = None
                        i_pos = None
                    
                    if pos is not None and i_pos is not None:
                        eaf.tiers[eng_pos_tiername_speaker][1][aid] = (shg_morphs_items[i][0], i_pos, None, None)

                        if (re.match("< ", pos)) and (i > 0):
                            pos = pos[2:]
                            prev_id = list_of_aids[-1]
                            if eaf.tiers[eng_pos_tiername_speaker][1][prev_id][1] is None:
                                eaf.tiers[eng_pos_tiername_speaker][1][prev_id] = (
                                    eaf.tiers[eng_pos_tiername_speaker][1][prev_id][0],
                                    pos, None, None
                                )

                            if re.match("\?", pos):
                                s_counter += 1
                            else:
                                g_counter += 1

                        list_of_aids.append(aid)

                print(f"> Проставлены POS-теги "
                      f"(теги проставлены для {g_counter} морфем, "
                      f"варианты предложены для {s_counter} морфем).")
            
            
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
    
        t = re.sub(":", "", re.search("[^\.]+", datetime.datetime.now().isoformat())[0])
        pympi.Elan.to_eaf(f"{orig_filename}_auto_{t}.eaf", eaf)

        print(f"\nГотово! Итоговый файл называется '{orig_filename}_auto_{t}.eaf'.\n")
    
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
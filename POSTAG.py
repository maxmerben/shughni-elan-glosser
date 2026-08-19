import csv, os, pympi, re, datetime, TOKENIZE
import cortex as cx

pos_list_path = "pos_list.csv"

def get_pos_table():
    with (open(pos_list_path, "r", encoding="utf-8-sig") as f):
        reader = csv.reader(f, delimiter=",")

        pos_list = []

        for row in reader:
            pos_list.append({
                "gloss":     row[1],
                "morph":     row[0] if row[0] != "" else ".+",
                "pos":       row[2],
                "condition": "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
            })
    return pos_list


def postag_text(glossed_filename: str):

    pos_list = get_pos_table()
    glossed_filename = re.sub(".eaf$", "", glossed_filename)

    try:
        text = cx.Text(filename=glossed_filename+".eaf")

        g_counter, s_counter = 0, 0
        
        for token in text.tokens:
            if not token:
                continue
            for morpheme in token[0]:
                for pos in pos_list:
                    if not re.fullmatch(pos["morph"], morpheme.morph):
                        continue
                    if not re.fullmatch(pos["gloss"], morpheme.gloss):
                        continue
                    morpheme.pos = pos["pos"]
                    if re.match(r"\?", pos["pos"]):
                        s_counter += 1
                    else:
                        g_counter += 1
                    break
        
        print(f"> Проставлены POS-теги "
            f"(теги проставлены для {g_counter} морфем, "
            f"варианты предложены для {s_counter} морфем).")
        
        t = re.sub(":", "", re.search("[^\.]+", datetime.datetime.now().isoformat())[0])
        short = re.search("(^.+)_auto", glossed_filename)
        if short is None:
            new_filename = f"{glossed_filename}_POS_{t}.eaf"
        else:
            new_filename = f"{re.sub('_auto.+', '', glossed_filename)}_POS_{t}.eaf"
        text.to_eaf(filepath=new_filename, suffix="", update_eaf=True)

        print(f"\nГотово! Итоговый файл называется '{new_filename}'.\n")

        return new_filename
    
    except FileNotFoundError:
        print("\nОШИБКА! Такой eaf-файл не найден.\n")


def find_and_postag_text(filename=None, interface=False):
    
    if interface:
        i = 0
        for f in os.listdir():
            if f.endswith(".eaf") and (re.search("auto", f) is None):
                i += 1
                x = f
        
        if i == 1:
            glossed_filename = x
        
        else:
            glossed_filename = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                input("Введите название файла .eaf: "))
    
        postag_text(glossed_filename)
    
    elif filename is not None and isinstance(filename, str):
        glossed_filename = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), filename)
        
        postag_text(glossed_filename)


if __name__ == "__main__":
    
    while True:
        
        find_and_postag_text(interface=True)
        
        goon = input("…Ещё один файл? Нажмите Enter.\nЕсли хотите выйти, напечатайте что угодно. ")
        if goon!="":
            break
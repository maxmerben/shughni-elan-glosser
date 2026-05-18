import csv, os, pympi, re, datetime, TOKENIZE
import cortex as cx


def get_context(token):
    a = max(token.back_sent[0]._id, token._id - 3)
    b = min(token.back_sent[-1]._id, token._id + 3)
    prefix, suffix = "", ""
    if a != token.back_sent[0]._id:
        prefix = "~"
    if b != token.back_sent[-1]._id:
        suffix = "~"
    context = []
    for t in token.back_sent:
        if a <= t._id <= b:
            context.append(t.token)
    return "'" + prefix + " ".join(context) + suffix + "'"



def check_text(filename):
    filename = re.sub(".eaf$", "", filename)

    text = cx.Text(filename=filename+".eaf")

    tokenization_incorrect_list = []
    morphemization_incorrect_list = []
    question_signs_list = []
    empty_gloss_list = []
    empty_pos_list = []

    for sentence in text.sentences:
        if sentence.tokenization_is_correct(quiet=True) is not True:
            tokenization_incorrect_list.append(sentence)
    
    for token in text.tokens:
        if token.morphemization_is_correct() is not True:
            morphemization_incorrect_list.append(token)

    for morpheme in text.morphemes:
        if isinstance(morpheme.gloss, str) and (re.search("^\?+ +.+", morpheme.gloss) or re.search("/ | /", morpheme.gloss)):
            question_signs_list.append(morpheme)
        elif isinstance(morpheme.pos, str) and (re.search("^\?+ +.+", morpheme.pos) or re.search("/ | /", morpheme.pos)):
            question_signs_list.append(morpheme)
        if morpheme.gloss in ("", "_", None):
            empty_gloss_list.append(morpheme)
        elif morpheme.pos in ("", None):
            empty_pos_list.append(morpheme)
    
    if len(tokenization_incorrect_list) > 0:
        print(f"\n> Есть предложения, в которых слой с текстом и слой с токенами не совпадают ({len(tokenization_incorrect_list)})!")
        for s in tokenization_incorrect_list:
            print(f"  > предложение: '{s.text}'")
            print(f"    токены:      '{' '.join([t.token for t in s.tokens])}'")
    
    if len(morphemization_incorrect_list) > 0:
        print(f"\n> Есть токены, у которых морфемы не соответствуют тексту токена ({len(morphemization_incorrect_list)})!")
        for t in morphemization_incorrect_list:
            print(f"  > токен (без пунктуации): '{cx.depunct(t.token)}'   \t(контекст: {get_context(t)})")
            print(f"    морфемы:                '{t.ana[0].morph_string}'")
    
    if len(question_signs_list) > 0:
        print(f"\n> Есть морфемы, где вы забыли убрать знак вопроса или слэш ({len(question_signs_list)})!")
        for m in question_signs_list:
            gloss = m.gloss if m.gloss != "_" else ""
            pos = m.pos if m.pos is not None else ""
            print(f"  > морфема: '{m.morph}'   \t(контекст: {get_context(m.back_token)})")
            print(f"    глосса:  '{gloss}'\n    POS-тег: '{pos}'")
    
    if len(empty_gloss_list) > 0:
        print(f"\n> Есть морфемы с пустыми глоссами ({len(empty_gloss_list)})!")
        for m in empty_gloss_list:
            gloss = m.gloss if m.gloss != "_" else ""
            pos = m.pos if m.pos is not None else ""
            print(f"  > морфема: '{m.morph}'   \t(контекст: {get_context(m.back_token)})")
            print(f"    глосса:  '{gloss}'\n    POS-тег: '{pos}'")
    
    if len(empty_pos_list) > 0:
        print(f"\n> Есть морфемы с пустыми POS-тегами ({len(empty_pos_list)})!")
        for m in empty_pos_list:
            if m not in empty_gloss_list:
                gloss = re.sub("_", "", m.gloss)
                pos = m.pos if m.pos is not None else ""
                print(f"  > морфема: '{m.morph}'   \t(контекст: {get_context(m.back_token)})")
                print(f"    глосса:  '{gloss}'\n    POS-тег: '{pos}'")

    print()

    if len(tokenization_incorrect_list + morphemization_incorrect_list + question_signs_list + empty_gloss_list + empty_pos_list) == 0:
        print("Всё отлично! Ура!\n\n:)\n")


def find_and_check_text(filename=None, interface=False):
    
    if interface:
        i = 0
        for f in os.listdir():
            if f.endswith(".eaf") and (re.search("auto", f) is None):
                i += 1
                x = f
        
        if i == 1:
            filename_to_check = x
        
        else:
            filename_to_check = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                input("Введите название файла .eaf: "))
    
        check_text(filename_to_check)
    
    elif filename is not None and isinstance(filename, str):
        filename_to_check = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), filename)
        
        check_text(filename_to_check)


if __name__ == "__main__":
    
    while True:
        
        find_and_check_text(interface=True)
        
        goon = input("…Ещё один файл? Нажмите Enter.\nЕсли хотите выйти, напечатайте что угодно. ")
        if goon!="":
            break
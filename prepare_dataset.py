import fontforge
import os
import csv


def is_acceptable(uni):
    if ord('a') <= uni <= ord('z'):
        return True

    if ord('A') <= uni <= ord('Z'):
        return True

    return ord('0') <= uni <= ord('9')


def extract_splineset(source, symb):
    tmp = source[source.find("StartChar: " + symb + '\n'):]
    return tmp[tmp.find('SplineSet'):tmp.find('EndSplineSet') + len('EndSplineSet')].replace('\n', '\\n')


# global id
global_glyph_id = 1

raw_train_dataset = open('data_train', 'w')
raw_test_dataset = open('data_test', 'w')

with open('glyphazzn_urls.txt') as dataset_list:
    dataset_reader = csv.reader(dataset_list, delimiter=',')
    for row in dataset_reader:
        glyph_id, glyph_type, glyph_url = [x.strip() for x in row]

        try:
            font_name = glyph_url.split('/')[-1]

            font = fontforge.open(glyph_type + "/" + font_name)
            font.save('tmp.sfd')
            raw_font = open('tmp.sfd', 'r').read()

            for symb in [chr(x) for x in range(ord('a'), ord('z') + 1)] + \
                    [chr(x) for x in range(ord('A'), ord('Z') + 1)] + \
                    ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']:
                glyph = font[symb]
                if is_acceptable(glyph.unicode):
                    if glyph_type == 'train':
                        print(global_glyph_id, glyph.unicode, glyph.width, glyph.vwidth
                            , extract_splineset(raw_font, symb), glyph_id, sep='$', file=raw_train_dataset)
                    else:
                        print(global_glyph_id, glyph.unicode, glyph.width, glyph.vwidth
                            , extract_splineset(raw_font, symb), glyph_id, sep='$', file=raw_test_dataset)
                    global_glyph_id += 1
        except Exception as e:
            print(e)
            pass

raw_train_dataset.close()
raw_test_dataset.close()

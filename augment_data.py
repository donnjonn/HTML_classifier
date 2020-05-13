import csv
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw


CSV_READ = 'data.csv'
CSV_WRITE = 'data_augment.csv'
AUG_AMOUNT = 200

def augment_data(csv_read, csv_write, aug_amount):
    with open(csv_read,'r') as datar, open(csv_write, 'w', newline='') as dataw:
        w = csv.writer(dataw, delimiter="\t")
        for row in csv.reader(datar, delimiter='\t'):
            print(row)
            w.writerow(row)
            text = row[1]
            for i in range(aug_amount):
                aug = nac.RandomCharAug(action="substitute", aug_char_p=0.2, aug_char_max=10000000)
                augmented_text = aug.augment(text)
                aug = naw.RandomWordAug(action="swap", aug_max=100)
                augmented_text = aug.augment(augmented_text)
                w.writerow([row[0], augmented_text])
                
def main():
    augment_data(CSV_READ, CSV_WRITE, AUG_AMOUNT)

if __name__ == "__main__":
    main()
                
  
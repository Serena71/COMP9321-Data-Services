import pandas as pd


def read_csv(csv_file):
    """
    :param csv_file: the path of csv file
    :return: A dataframe out of the csv file
    """
    return pd.read_csv(csv_file)


def write_in_csv(dataframe, file):
    """
    :param dataframe: The dataframe which must be written into a csv file
    :param file: where the csv must be stored
    """
    dataframe.to_csv(file, sep=',', encoding='utf-8')


def print_dataframe(dataframe, print_column=True, print_rows=True):
    # print column names
    if print_column:
        print(','.join(dataframe.columns))

    # print rows one by one
    if print_rows:
        for row in dataframe.itertuples(index=False, name=None):
            row = ','.join(str(col) for col in row)
            print(row)

if __name__ == '__main__':
    csv_file = 'Demographic_Statistics_By_Zip_Code.csv'  # path to the downloaded csv file
    dataframe = read_csv(csv_file)

    print("Columns of csv file")
    print_dataframe(dataframe, True, False)
    print("Rows of csv file")
    print_dataframe(dataframe, False, True)



    #print("Write the dataframe as a csv file")
    #write_in_csv(dataframe, "Demographic_Statistics_New.csv")  # path where the new csv file is stored

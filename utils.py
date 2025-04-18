import opusFC

def get_Opus_data(filename):
    dbs = opusFC.listContents(filename)
    data = opusFC.getOpusData(filename, dbs[0])
    return data.spectra, data.x

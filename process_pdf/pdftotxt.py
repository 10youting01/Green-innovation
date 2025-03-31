import os
from io import StringIO
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage


#建立convert函數
#output:管理處理後的輸出
#manager:是一個pdf資源管理器，目的是為了儲存共享資源
#converter:主要處理轉換的地方
#interpreter:創建一個pdf解析器
def convert(fname, pages=None):
    if not pages:
        pagenums = set()
    else:
        pagenums = set(pages)

    output = StringIO()
    manager = PDFResourceManager()
    converter = TextConverter(manager, output, laparams=LAParams())
    interpreter = PDFPageInterpreter(manager, converter)
    infile = open(fname, 'rb')
    for page in PDFPage.get_pages(infile, pagenums):
        interpreter.process_page(page)
    infile.close()
    converter.close()
    text = output.getvalue()
    output.close
    return text


def convertMultiple(pdfDir, txtDir):
    #iterate through pdfs in pdf directory
    for pdf in os.listdir(pdfDir): 
        fileExtension = pdf.split(".")[-1]
        if fileExtension == "pdf":
            pdfFilename = os.path.join(pdfDir, pdf)  # 修正路徑拼接問題
            #get string of text content of pdf
            text = convert(pdfFilename) 
            textFilename = os.path.join(txtDir, pdf[:-4] + ".txt")  # 同樣修正 txt 檔案輸出路徑
            #make text file
            with open(textFilename, "w", encoding='utf-8') as textFile:
                textFile.write(text) 
            print("Finished converting to txt:", pdf)
    print("Finished converting all files")

    
# =============================================================================
# for pdf in os.listdir(pdfDir): 
#     fileExtension = pdf.split(".")[-1]
#     if fileExtension == "pdf":
#         pdfFilename = pdfDir + pdf
# =============================================================================
            
pdfdir=os.path.abspath('/home/francia/research_hub/csr_project/CSR Reporting/NYSE/1998/')
textdir=os.path.abspath('/home/francia/research_hub/csr_project/test_output/')
convertMultiple(pdfdir,textdir)
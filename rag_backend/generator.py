from llm_model import model
from prompt import rules_prompt
from prompt import parser

def generator(query : str, context: str):

    response = rules_prompt | model | parser

    result = response.invoke({"user_query" : query, "response_doc" : context})

    return result

if __name__ == "__main__":
    result = generator(query = "Who is defined as a Data Fiduciary and how does that role differ from a Data Processor when it comes to responsibility for complying with the Act?",
                       context =  [
      "(h) “data” means a representation of information, facts, concepts, opinions or instructions in a manner suitable for communication, interpretation or processing by human beings or by automated means; (i) “Data Fiduciary” means any person who alone or in conjunction with other persons determines the purpose and means of processing of personal data; (j) “Data Principal” means the individual to whom the personal data relates and where such individual is— (i) a child, includes the parents or lawful guardian of such a child; (ii) a person with disability, includes her lawful guardian, acting on her behalf; (k) “Data Processor” means any person who processes personal data on behalf of a Data Fiduciary; (l) “Data Protection Officer” means an individual appointed by the Significant Data Fiduciary under clause (a) of sub-section (2) of section 10; (m) “digital office” means an office that adopts an online mechanism wherein the proceedings, from receipt of intimation or complaint or reference or directions or appeal, as the case may be, to the disposal thereof, are conducted in online or digital mode; (n) “digital personal data” means personal data in digital form; (o) “gain” means— (i) a",
      "to carry out the duties provided under this Act, be responsible for complying with the provisions of this Act and the rules made thereunder in respect of any processing undertaken by it or on its behalf by a Data Processor. (2) A Data Fiduciary may engage, appoint, use or otherwise involve a Data Processor to process personal data on its behalf for any activity related to offering of goods or services to Data Principals only under a valid contract. (3) Where personal data processed by a Data Fiduciary is likely to be— (a) used to make a decision that affects the Data Principal; or (b) disclosed to another Data Fiduciary, the Data Fiduciary processing such personal data shall ensure its completeness, accuracy and consistency. (4) A Data Fiduciary shall implement appropriate technical and organisational measures to ensure effective observance of the provisions of this Act and the rules made thereunder. (5) A Data Fiduciary shall protect personal data in its possession or under its control, including in respect of any processing undertaken by it or on its behalf by a Data Processor, by taking reasonable security safeguards to prevent personal data breach. (6) In the event of a"
    ])

    print(result)
from pydantic import BaseModel

class FinalReport(BaseModel):
    content: str
    sources: list[str]


class FinalReportWrapper(BaseModel):
    report: FinalReport
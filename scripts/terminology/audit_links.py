"""Human/agent judgments of the fixed 100-link seed-13 sample, dictionary only."""
import json
from termmap.common.io import load_config,term_dir,write_json

# Each item is an individual review of link_audit_sample.json, in saved order.
JUDGMENTS = [
    (True,"Specific method to method"), (True,"Entire structure to general structure"),
    (False,"Arterial wall is part of artery, not a subtype"), (True,"Specific vertebral body"),
    (False,"Type-of-vein wording is an abstract category"), (False,"Combined specimen loses organs and adds a side"),
    (True,"Specific anatomical bursa"), (True,"Entire capsule to capsule structure"),
    (False,"Cavity to joint and unsupported left side"), (True,"Specific deep vein"),
    (True,"Laterality siblings"), (False,"Different embryo stage used as parent"),
    (True,"Stage qualifier siblings"), (False,"Lymphatics are not a fallopian tube"),
    (False,"Nail plate is part of finger"), (True,"Specific lymphatic follicle"),
    (False,"Prostate region is part of prostate"), (True,"Specific esophageal third"),
    (True,"Laterality siblings"), (True,"Specific osteochondritis"),
    (True,"Allergic subtype of contact dermatitis"), (False,"Unsupported acute qualifier"),
    (True,"Specific viral arthritis"), (True,"Specific radiation disorder"),
    (True,"Infarction qualifier siblings"), (True,"Specific enteropathic arthritis"),
    (True,"Site-specific melanoma"), (True,"Specific metacarpal fracture"),
    (True,"Benign neoplasm to neoplasm"), (True,"Laterality siblings"),
    (False,"Parent introduces unsupported left side"), (True,"Specific sickle cell disease"),
    (True,"Self-induced cheilitis is a factitious disorder"), (True,"Specific tibial fracture"),
    (True,"Specific neuropathy site"), (True,"Specific inhibitor response"),
    (True,"Discoid lupus belongs to chronic discoid lupus"), (False,"Nevus is not an acute disorder"),
    (True,"Specific pulmonary edema"), (False,"Chronic to acute is contradictory"),
    (True,"Laterality-specific pinguecula"), (False,"Unsupported right forearm qualifier"),
    (False,"Hair sinus does not establish a hair-disorder parent"), (True,"Laterality siblings"),
    (True,"Site-specific incomplete ossification"), (True,"Specific infection"),
    (False,"Infective does not establish chronic mastitis"), (True,"Site-specific superficial injury"),
    (True,"Combined injury includes tendon disorder"), (True,"Specific place of accident"),
    (False,"Different Gleason score used as parent"), (False,"Different acuity value and side"),
    (True,"Specific calcitonin allergy"), (True,"Laterality siblings"),
    (False,"Different action and assertion"), (False,"High pain threshold does not imply pain"),
    (True,"Inflammation is a joint finding"), (False,"Sensation is not absence of joint"),
    (False,"Only-product parent excludes the second ingredient"), (True,"Contains propranolol"),
    (True,"Contains neomycin"), (False,"Aleukemic lymphoid does not imply chronic leukemia"),
    (True,"Virus taxonomic group"), (False,"Different subspecies is not a parent"),
    (True,"Specific Streptococcus"), (True,"Specific fluke"), (True,"Specific Hydrotaea"),
    (True,"Specific Salmonella"), (True,"Specific antigen assay"), (True,"Specific surgical approach"),
    (True,"Site-specific biopsy"), (True,"Specific interposition procedure"),
    (True,"Laterality qualifier siblings"), (True,"Laterality siblings"),
    (True,"Specific ear biopsy"), (True,"Tongue biopsy is a tongue procedure"),
    (True,"Specific anatomical site"), (True,"Specific removal procedure"),
    (True,"Specific destruction procedure"), (True,"Combined procedure includes fitting"),
    (True,"Combined procedure includes drainage"), (True,"Specific plastic repair"),
    (True,"Laterality siblings"), (True,"Phenothiazine derivative product family"),
    (True,"Grade qualifier siblings"), (True,"Specific allowable value"),
    (False,"Increment is a distinct observable from concentration"), (True,"Assertion qualifier siblings"),
    (True,"Specific imaging projection"), (False,"Vaccination does not imply suspected disease"),
    (True,"Specific interleukin"), (False,"Enzyme is not its substrate"),
    (True,"Specific salt hydration form"), (False,"Different antibody specificity"),
    (True,"Type qualifier siblings"), (False,"Different blood group antigen"),
    (False,"Different blood group antigen"), (True,"Specific food"),
    (True,"Specific dehydrogenase"), (False,"Different hemoglobin variant"),
]


def main():
    folder=term_dir(load_config())
    if folder.name!="9ba6e32a296594c0672d":
        raise RuntimeError("These manual judgments apply only to the reviewed dictionary release; review a new sample")
    path=folder/"link_audit_sample.json"
    sample=json.loads(path.read_text())
    assert len(sample)==len(JUDGMENTS)==100
    for row,(correct,reason) in zip(sample,JUDGMENTS):
        row["judgment"]="correct" if correct else "wrong";row["reason"]=reason
    write_json(path,sample)
    print("Manual link precision:",sum(c for c,_ in JUDGMENTS),"/ 100")


if __name__=="__main__":main()

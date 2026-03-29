-- =============================================================================
-- Diabetes 130-US Hospitals Dataset — Normalized Schema (3NF)
-- =============================================================================

PRAGMA foreign_keys = ON;

-- -----------------------------------------------------------------------------
-- Lookup: Admission Types
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS admission_types (
    id          INTEGER PRIMARY KEY,
    description TEXT    NOT NULL
);

-- -----------------------------------------------------------------------------
-- Lookup: Discharge Types
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS discharge_types (
    id          INTEGER PRIMARY KEY,
    description TEXT    NOT NULL
);

-- -----------------------------------------------------------------------------
-- Lookup: ICD-9 Diagnosis Codes
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS diagnoses_lookup (
    icd9_code   TEXT    PRIMARY KEY,
    description TEXT,
    category    TEXT
);

CREATE INDEX IF NOT EXISTS idx_diagnoses_lookup_category ON diagnoses_lookup(category);

-- -----------------------------------------------------------------------------
-- Patients (one row per unique patient)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS patients (
    patient_id  INTEGER PRIMARY KEY,
    race        TEXT,
    gender      TEXT,
    age_group   TEXT
);

CREATE INDEX IF NOT EXISTS idx_patients_age_group ON patients(age_group);
CREATE INDEX IF NOT EXISTS idx_patients_race       ON patients(race);
CREATE INDEX IF NOT EXISTS idx_patients_gender     ON patients(gender);

-- -----------------------------------------------------------------------------
-- Admissions / Encounters (one row per hospital encounter)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS admissions (
    encounter_id        INTEGER PRIMARY KEY,
    patient_id          INTEGER NOT NULL REFERENCES patients(patient_id),
    admission_type_id   INTEGER REFERENCES admission_types(id),
    discharge_type_id   INTEGER REFERENCES discharge_types(id),
    time_in_hospital    INTEGER,
    num_lab_procedures  INTEGER,
    num_procedures      INTEGER,
    num_medications     INTEGER,
    num_diagnoses       INTEGER,
    hba1c_result        TEXT,
    change_medications  TEXT,
    diabetes_medication TEXT,
    readmission         TEXT
);

CREATE INDEX IF NOT EXISTS idx_admissions_patient_id        ON admissions(patient_id);
CREATE INDEX IF NOT EXISTS idx_admissions_admission_type_id ON admissions(admission_type_id);
CREATE INDEX IF NOT EXISTS idx_admissions_discharge_type_id ON admissions(discharge_type_id);
CREATE INDEX IF NOT EXISTS idx_admissions_readmission       ON admissions(readmission);
CREATE INDEX IF NOT EXISTS idx_admissions_hba1c_result      ON admissions(hba1c_result);

-- -----------------------------------------------------------------------------
-- Diagnosis Encounters (bridge: admissions ↔ diagnoses_lookup)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS diagnosis_encounters (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    encounter_id        INTEGER NOT NULL REFERENCES admissions(encounter_id),
    icd9_code           TEXT    REFERENCES diagnoses_lookup(icd9_code),
    diagnosis_position  INTEGER NOT NULL   -- 1=primary, 2=secondary, 3=additional
);

CREATE INDEX IF NOT EXISTS idx_diag_enc_encounter_id ON diagnosis_encounters(encounter_id);
CREATE INDEX IF NOT EXISTS idx_diag_enc_icd9_code    ON diagnosis_encounters(icd9_code);
CREATE INDEX IF NOT EXISTS idx_diag_enc_position     ON diagnosis_encounters(diagnosis_position);

-- -----------------------------------------------------------------------------
-- Medications per Encounter
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS medications (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    encounter_id     INTEGER NOT NULL REFERENCES admissions(encounter_id),
    drug_name        TEXT    NOT NULL,
    prescribed       TEXT,           -- original value: 'Steady', 'Up', 'Down', 'No'
    change_indicator TEXT            -- 'Ch' = changed, 'No' = no change
);

CREATE INDEX IF NOT EXISTS idx_medications_encounter_id ON medications(encounter_id);
CREATE INDEX IF NOT EXISTS idx_medications_drug_name    ON medications(drug_name);

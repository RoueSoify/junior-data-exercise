from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import ArrayType, StringType

def main():
    # Initialisation de Spark
    spark = SparkSession.builder.appName("PatientFHIR_Pipeline").getOrCreate()

    # Chargement des données (en supposant que le dossier resources/ est à côté du script)
    df_patients = spark.read.csv("resources/patients.csv", header=True, inferSchema=True)
    df_ipp = spark.read.csv("resources/identifiants_ipp.csv", header=True, inferSchema=True)
    df_adresses = spark.read.csv("resources/adresses.csv", header=True, inferSchema=True)
    df_opp = spark.read.csv("resources/opposition_recherche.csv", header=True, inferSchema=True)

    # Résolution des IPP
    # On crée un mapping : tout IPP déprécié est redirigé vers son IPP principal
    df_mapping_ipp = df_ipp.select(
        F.col("ipp").alias("ipp_source"),
        F.coalesce(F.col("ipp_principal"), F.col("ipp")).alias("ipp_actif")
    )

    # Fonction utilitaire pour appliquer le mapping d'IPP sur n'importe quel dataframe
    def align_ipp(df):
        return df.join(df_mapping_ipp, df.ipp == df_mapping_ipp.ipp_source, "left") \
                 .withColumn("ipp_final", F.coalesce(F.col("ipp_actif"), F.col("ipp"))) \
                 .drop("ipp", "ipp_source", "ipp_actif") \
                 .withColumnRenamed("ipp_final", "ipp")

    # Nettoyage des patients
    df_patients_clean = align_ipp(df_patients)
    
    # Nettoyage des chaînes et dédoublonnage (ex: espaces en trop)
    df_patients_clean = df_patients_clean.withColumn("nom_naissance", F.trim(F.col("nom_naissance"))) \
                                         .dropDuplicates(["ipp", "nom_naissance"])

    # Standardisation des dates (gestion des multiples formats) (ISO 8601 pour la vie)
    date_formats = ['yyyy-MM-dd', 'dd/MM/yyyy', 'dd-MM-yyyy', 'yyyy/MM/dd']
    df_patients_clean = df_patients_clean.withColumn(
        "birthDate", 
        F.coalesce(*[F.try_to_date(F.col("date_naissance"), f) for f in date_formats])
    )

    # Standardisation du sexe pour FHIR
    df_patients_clean = df_patients_clean.withColumn(
        "gender",
        F.when(F.lower(F.col("sexe")).isin("m", "1", "homme", "male"), "male")
         .when(F.lower(F.col("sexe")).isin("f", "2", "femme", "female"), "female")
         .otherwise("unknown")
    )

    # Parsing des prénoms (format JSON textuel dans le CSV vers un vrai tableau Spark)
    df_patients_clean = df_patients_clean.withColumn(
        "prenoms_array", 
        F.from_json(F.col("prenoms"), ArrayType(StringType()))
    )

    # Création de la structure "name" au format FHIR
    df_patients_clean = df_patients_clean.withColumn(
        "name",
        F.array(
            F.struct(
                F.lit("official").alias("use"),
                F.col("nom_naissance").alias("family"),
                F.col("prenoms_array").alias("given")
            )
        )
    )

    # Nettoyage des Adresses
    df_adresses_clean = align_ipp(df_adresses)
    df_adresses_clean = df_adresses_clean.withColumn(
        "address_struct",
        F.struct(
            F.col("type_adresse").alias("use"), # Simplification: mapping direct
            F.col("ligne_adresse").alias("line"),
            F.col("ville").alias("city"),
            F.col("code_postal").alias("postalCode"),
            F.col("pays").alias("country")
        )
    )
    # Regrouper toutes les adresses par IPP
    df_adresses_grouped = df_adresses_clean.groupBy("ipp").agg(
        F.collect_list("address_struct").alias("address")
    )

    # Jointure Finale et Formatage FHIR Patient
    df_final = df_patients_clean.join(df_adresses_grouped, "ipp", "left")

    # Construction de l'objet final JSON
    df_fhir = df_final.select(
        F.lit("Patient").alias("resourceType"),
        F.array(F.struct(F.lit("IPP").alias("system"), F.col("ipp").cast("string").alias("value"))).alias("identifier"),
        F.col("name"),
        F.col("gender"),
        F.col("birthDate"),
        F.col("address")
    )

    # Écriture du résultat
    df_fhir.write.mode("overwrite").json("output_fhir")
    print("Traitement terminé, les données sont dans le dossier output_fhir/")

if __name__ == "__main__":
    main()

import determinar_opiniones
import os
import helpers


def lanzar():
    '''Lanza el menu
    '''

    print("========================")
    print('PREPARANDO...')
    print("========================")
    tweets = determinar_opiniones.Predicciones_cambio_climatico('datas/calentamientoClimatico.csv')
    tweets.preparacion()
    tweets.dividir_dataset()


    while True:
        helpers.limpiar_pantalla()
        print("========================")
        print(" BIENVENIDO AL Manager ")
        print(" Instrucciones: Primero seleccione un modelo y despues pruebe a introducir una frase ")
        print("========================")
        print("[1] Modelo bayesiano ")
        print("[2] Modelo SVM ")
        print("[3] Introducir frase ")
        print("[4] Salir ")
        print("========================")

        opcion = int(input("> "))
        helpers.limpiar_pantalla()

        if opcion == 1:
            tweets.aprendizaje_bayesiano()
            print(tweets)
            input("\n\nPress Enter to continue...")

        if opcion == 2:
            tweets.aprendizaje_svm()
            print(tweets)
            input("\n\nPress Enter to continue...")

        if opcion == 3:
            frase = input("Introduce la frase que quieres predecir:\n > ")
            if tweets.predecir_frase(frase):
                print(">> No cree en el calentamiento climático...")
            else:
                print(">> Cree en el calentamiento climático...")

            input("\n\nPress Enter to continue...")

        if opcion == 4:
            break


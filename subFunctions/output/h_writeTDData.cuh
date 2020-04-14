void h_writeTDData (
	paramsStruct params,
	observablesStruct observables,
	wfStruct psi
)
{
	FILE * IO;

	IO = fopen(LOCAL_TD_T, "wb");
    fwrite(params.h_t, sizeof(double), N_t, IO);
    fclose(IO);

	IO = fopen(LOCAL_TD_E, "wb");
    fwrite(params.h_E, sizeof(double), N_t, IO);
    fclose(IO);

	IO = fopen(LOCAL_TD_OBSERVABLES_DIPOLE_ACCELERATION, "wb");
    fwrite(observables.h_dipoleAcceleration, sizeof(double2), N_t, IO);
    fclose(IO);

	IO = fopen(LOCAL_TD_OBSERVABLES_NORM, "wb");
    fwrite(psi.h_norm, sizeof(double), N_t, IO);
    fclose(IO);
}

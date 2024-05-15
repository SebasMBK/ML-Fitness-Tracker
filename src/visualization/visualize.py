import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

def read_data(table_schema: str,
                table_name: str,
                username: str,
                password: str,
                hostname: str,
                port: int,
                database: str ) -> pd.DataFrame:
    conn_string = f"postgresql+psycopg2://{username}:{password}@{hostname}:{port}/{database}"
    engine = create_engine(conn_string)
    try:
        # Read table
        df = pd.read_sql_table(
            table_name=table_name,
            schema=table_schema,
            con=engine
        )
        df.index = df["epoch_ms"].astype('datetime64[ns]')
        # We want epoch_ms only in the index
        del df["epoch_ms"]
        engine.dispose()
    except SQLAlchemyError as e:
        raise SQLAlchemyError(f"An error ocurred while reading the data: {e}")
    finally:
        engine.dispose()
    return df

if __name__ == '__main__':

    df = read_data(
        table_schema="merged",
        table_name="fitness_tracker",
        username="postgres",
        password="postgres",
        hostname="localhost",
        port=5432,
        database="ml-fitness-tracker"
    )
    
    df_set_column = df[df["set"] == 1]
    plt.plot(df_set_column["y_axis_g"])
    
    # Adjusting plot settings
    mpl.style.use("classic")
    mpl.rcParams["figure.figsize"] = (15,5)
    mpl.rcParams["figure.dpi"] = 100

    for label in df["label"].unique():
        subset = df[df["label"] == label]
        fig, ax = plt.subplots()
        plt.plot(subset["y_axis_g"].reset_index(drop=True), label=label)
        plt.legend()
        plt.show()

    # 100 examples for each exercise
    for label in df["label"].unique():
        subset = df[df["label"] == label]
        fig, ax = plt.subplots()
        plt.plot(subset[:100]["y_axis_g"].reset_index(drop=True), label=label)
        plt.legend()
        plt.show()
    
    # Compare medium vs heavy sets for subject A doing squats
    # We can tell by the acceleration that the subject moves faster
    # in medium sets than in heavy sets
    df_squats = df.query("label == 'squat'").query("participant == 'A'").reset_index(drop=True)
    fig, ax = plt.subplots()
    df_squats.groupby(["category"])["y_axis_g"].plot()
    ax.set_ylabel('y_axis_g')
    ax.set_xlabel('Samples')
    plt.legend()

    # Comparing participants
    # Sorting is important, otherwise, we would get a very messy plot
    df_participant = df.query("label == 'bench'").sort_values("participant").reset_index(drop=True)
    fig, ax = plt.subplots()
    df_participant.groupby(["participant"])["y_axis_g"].plot()
    ax.set_ylabel('y_axis_g')
    ax.set_xlabel('Samples')
    plt.legend()

    # Plotting multiple axis (x, y and z)
    label = "squat"
    participant = "A"
    df_all_axis = df.query(f"label == '{label}'").query(f"participant == '{participant}'").reset_index(drop=True)
    fig, ax = plt.subplots()
    df_all_axis[["x_axis_g", "y_axis_g", "z_axis_g"]].plot(ax=ax)
    ax.set_ylabel('axis_g')
    ax.set_xlabel('Samples')
    plt.legend()

    # Get all exercises for all participants for both sensors (accelerometer and gyroscope)
    labels = df["label"].unique()
    participants = df["participant"].unique()

    # Accelerometer
    for label in labels:
        for participant in participants:
            df_all_axis = df.query(f"label == '{label}'")\
                            .query(f"participant == '{participant}'")\
                            .reset_index(drop=True)
            fig, ax = plt.subplots()
            df_all_axis[["x_axis_g", "y_axis_g", "z_axis_g"]].plot(ax=ax)
            ax.set_ylabel('axis_g')
            ax.set_xlabel('Samples')
            plt.title(f"{label} ({participant})")
            plt.legend()
    
    # Gyroscope
    for label in labels:
        for participant in participants:
            df_all_axis = df.query(f"label == '{label}'")\
                            .query(f"participant == '{participant}'")\
                            .reset_index(drop=True)
            fig, ax = plt.subplots()
            df_all_axis[["x_axis_deg_s", "y_axis_deg_s", "z_axis_deg_s"]].plot(ax=ax)
            ax.set_ylabel('axis_deg_s')
            ax.set_xlabel('Samples')
            plt.title(f"{label} ({participant})")
            plt.legend()
    
    # Get both sensor data into 1 figure
    labels = df["label"].unique()
    participants = df["participant"].unique()

    for label in labels:
        for participant in participants:
            df_combined_sensors = df.query(f"label == '{label}'")\
                                    .query(f"participant == '{participant}'")\
                                    .reset_index(drop=True)
            if len(df_combined_sensors) > 0:
                fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20,10))
                df_combined_sensors[["x_axis_g", "y_axis_g", "z_axis_g"]].plot(ax=ax[0])
                df_combined_sensors[["x_axis_deg_s", "y_axis_deg_s", "z_axis_deg_s"]].plot(ax=ax[1])
                ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True)
                ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True)
                ax[1].set_xlabel('Samples')

                plt.savefig(f"../../reports/figures/{label.title()} ({participant}).png")
                plt.show()



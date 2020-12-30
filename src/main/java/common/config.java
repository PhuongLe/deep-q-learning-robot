package common;

import java.text.SimpleDateFormat;
import java.util.Date;

public class config {
    public static String BaseFolder = "d:\\Google Drive\\LXP\\UBC\\Term 3\\CPEN 502 - ML\\Assignments\\Robocode\\out\\report\\" + new SimpleDateFormat("yyyy-MM-dd-HH-mm-ss") .format(new Date());
    public static String LogFileName = BaseFolder+ "-neural-net.log";

}

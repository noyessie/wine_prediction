package com.njit.edu.cs643.v2.helper;

import java.util.Date;

public class Utils {

    public static String getTimestamp(){
        return new Date().getTime()+"";
    }
}

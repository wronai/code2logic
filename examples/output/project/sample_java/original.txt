// Sample Java file for multi-language reproduction tests.
// Focus: classes, interfaces, enums, records, and static utility methods.

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;

public class SampleJava {

    public enum Status {
        NEW,
        ACTIVE,
        DISABLED
    }

    public interface Identifiable {
        String getId();
    }

    public record User(String id, String name, Status status) implements Identifiable {
        @Override
        public String getId() {
            return id;
        }
    }

    public static final int DEFAULT_LIMIT = 10;

    public static List<User> filterActive(List<User> users) {
        if (users == null) {
            return List.of();
        }
        List<User> out = new ArrayList<>();
        for (User u : users) {
            if (u != null && u.status() == Status.ACTIVE) {
                out.add(u);
            }
        }
        return out;
    }

    public static List<User> sortByName(List<User> users) {
        if (users == null) {
            return List.of();
        }
        List<User> copy = new ArrayList<>(users);
        copy.removeIf(Objects::isNull);
        copy.sort(Comparator.comparing(User::name, String.CASE_INSENSITIVE_ORDER));
        return Collections.unmodifiableList(copy);
    }
}

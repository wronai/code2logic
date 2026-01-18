// Sample C# file for multi-language reproduction tests.
// Focus: interfaces, records, classes, and simple business logic.

using System;
using System.Collections.Generic;
using System.Linq;

public interface IHasId
{
    string Id { get; }
}

public record User(string Id, string Name, bool IsActive) : IHasId;

public static class UserService
{
    public const int DefaultLimit = 10;

    public static List<User> FilterActive(IEnumerable<User> users)
    {
        if (users == null)
        {
            return new List<User>();
        }
        return users.Where(u => u != null && u.IsActive).ToList();
    }

    public static User FindById(IEnumerable<User> users, string id)
    {
        if (users == null || string.IsNullOrEmpty(id))
        {
            return null;
        }
        return users.FirstOrDefault(u => u != null && u.Id == id);
    }
}
